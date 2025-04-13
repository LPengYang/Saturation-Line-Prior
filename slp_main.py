import torch
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.utils as vutils
import argparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def synthesize_fog(J, t, A=None):
    """
    Synthesize hazy image base on optical model
    I = J * t + A * (1 - t)
    """

    if A is None:
        A = 1

    return J * t + A * (1 - t)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class GuidedFilter(torch.nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):    # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """
        
        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter( torch.ones(p.size()) )

        if I.is_cuda:
            N = N.cuda()

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I*p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I*I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b   


class restore_SLP(torch.nn.Module):
    def __init__(self,patch_size=15):
        super(restore_SLP,self).__init__()
        self.width_dcp = 55
        self.bili = 0.001
        self.maxpool = torch.nn.MaxPool2d(kernel_size=self.width_dcp,stride=1)
        self.patch_size = patch_size
        self.guidedfilter = GuidedFilter(r=2*patch_size, eps=1e-3)
    
    def get_dark_channel(self,x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width_dcp//2, self.width_dcp//2,self.width_dcp//2, self.width_dcp//2), mode='constant', value=1)
        x = -(self.maxpool(-x))
        return x

    def obtain_t_bccr(self,haze,map_A,patch_size):
        # haze = (haze+1)/2
        # map_A = (map_A +1)/2
        t_1 = (map_A-haze)/(map_A-20/255)
        t_2= (map_A-haze)/(map_A-300/255)
        t = torch.cat((t_1,t_2),dim=1)
        t,_ = torch.max(t, dim=1, keepdim=True) 
        t = F.pad(t, pad=(patch_size//2, patch_size//2, patch_size//2, patch_size//2), mode='constant', value=0)
        t = F.max_pool2d(t, kernel_size=patch_size, stride=1)

        return t.clamp(0.05,1)

    def get_map_A(self,x):
        dark_channel = self.get_dark_channel(x)
        searchidx = torch.argsort(dark_channel.view(x.shape[0],-1), dim=1, descending=True)[:,:int(x.shape[2]*x.shape[3]*0.001)]
        x_reshape = x.view(x.shape[0],3,-1)
        searched = torch.gather(x_reshape,dim=2,index=searchidx.unsqueeze(1).repeat(1,3,1))
        A_final_pixel = torch.mean(searched,dim=2,keepdim=True)       
        map_A = A_final_pixel.unsqueeze(3).repeat(1,1,x.shape[2],x.shape[3])

        return map_A
    
    def get_percentile_value(self, tensor,bili=0.05):
        tensor_clamped = torch.where(tensor > 0, tensor, 10)
        tensor_flat = tensor_clamped.view(tensor.size(0), -1)
        
        sorted_elements, _ = torch.sort(tensor_flat, dim=1)

        num_elements = tensor_flat.size(1) 
        index = int(bili * num_elements)

        index = min(index, num_elements - 1)

        percentile_values = sorted_elements[:, index]
        
        return percentile_values.reshape(-1,1,1,1)

    def pad_and_reshape_torch(self,data, patch_size):
        N, _, H, W = data.size()
    
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        
        padding = [pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2]
        padded_data = F.pad(data, padding, mode='replicate')

        patches = padded_data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        patches = patches.contiguous().view(N, -1, patch_size * patch_size).permute(0, 2, 1)

        start_H = pad_H // 2
        start_W = pad_W // 2
        
        padding_H = padded_data.size()[2] // patch_size
        padding_W = padded_data.size()[3] // patch_size
        
        return patches, start_H, start_W, padding_H, padding_W

    def get_transmission(self, S, V_1, patch_size = 15, pixel_number=10, slp_length=0.1):
        #  S/V_1: shape (N,1,H,W) 
        X_axis,start_H, start_W, padding_H, padding_W = self.pad_and_reshape_torch(V_1, patch_size)
        Y_axis,_,_,_,_ = self.pad_and_reshape_torch(S, patch_size)
        delta_X = X_axis.unsqueeze(2) - X_axis.unsqueeze(1)  
        delta_Y = Y_axis.unsqueeze(2) - Y_axis.unsqueeze(1)  
        slope_pair = delta_Y/(delta_X+1e-8)
        slope_mask = ((slope_pair > -1) & (slope_pair < 0)).float()
        slope_count = slope_mask.sum(dim=2)  
        pixel_mask = (slope_count>=0.5*(patch_size**2)).float()
        pixel_count = pixel_mask.sum(dim=1,keepdim=True)
        mean_x = (X_axis*pixel_mask).sum(dim=1,keepdim=True)/(pixel_count+1e-5)
        mean_y = (Y_axis*pixel_mask).sum(dim=1,keepdim=True)/(pixel_count+1e-5)
        slp_k = ((X_axis*Y_axis*pixel_mask).sum(dim=1,keepdim=True)-pixel_count*mean_x*mean_y)/(((X_axis*pixel_mask)**2).sum(dim=1,keepdim=True)-pixel_count*mean_x**2 +1e-5)
        slp_b = mean_y - slp_k*mean_x
        transmission = 1+slp_k/(slp_b+1e-8)
        length = ((1+slp_k **2)**0.5)*((X_axis*pixel_mask).max(dim=1,keepdim=True)[0]-(X_axis+1e8*(1-pixel_mask)).min(dim=1,keepdim=True)[0])
        transmission_mask = ((pixel_count>pixel_number) & (length>slp_length) & (slp_k >-1)& (slp_k <0) & (transmission>1e-2) & (transmission<1-1e-2)).float()
        transmission = transmission*transmission_mask
        transmission = F.interpolate(transmission.reshape(-1,1,padding_H,padding_W), scale_factor=patch_size, mode='nearest')
        return transmission[:,:,start_H:(S.shape[2]+start_H),start_W:(S.shape[3]+start_W)]

    def forward(self,image):
        map_A = self.get_map_A(image)
        image_norm = image/map_A
        image_S = 1- image_norm.min(dim=1,keepdim=True)[0]/(image_norm.max(dim=1,keepdim=True)[0]+1e-8)
        image_V_1 = 1/(image_norm.max(dim=1,keepdim=True)[0]+1e-8)
        t_slp_1 = self.get_transmission(image_S, image_V_1, patch_size = self.patch_size, pixel_number=10, slp_length=0.1)
        t_slp_2 = self.get_transmission(image_S[:,:,self.patch_size//2:,self.patch_size//2:], image_V_1[:,:,self.patch_size//2:,self.patch_size//2:], patch_size = self.patch_size, pixel_number=10, slp_length=0.1)
        t_slp_2 =  F.pad(t_slp_2, [self.patch_size//2,0,self.patch_size//2,0], mode='constant', value=0)
        t_slp = torch.where((t_slp_1 != 0) & (t_slp_2 != 0), (t_slp_1 + t_slp_2) / 2, torch.where(t_slp_1 != 0, t_slp_1, t_slp_2))
        t_bl = self.obtain_t_bccr(image,map_A,self.patch_size)
        t_min = self.get_percentile_value(t_slp)
        t_fusion = (torch.where(t_slp>0,t_slp,t_bl)).clamp(min=t_min,max=torch.ones_like(t_min))
        t_refined = self.guidedfilter(image.mean(dim=1,keepdim=True),t_fusion)

        dehaze_image = torch.div(image-map_A,t_refined)+map_A
        return dehaze_image

class MyDataSet_single(Dataset):
    def __init__(self, dataset_path_list, transform=None):
        """
        dataset_type: ['train', 'test']
        """
        self.transform = transform
        self.dir_I = os.path.join(dataset_path_list) 
        self.I_paths = sorted(make_dataset(self.dir_I))
        self.I_size = len(self.I_paths) 
        
    def __getitem__(self, index):
        I_path = self.I_paths[index % self.I_size]
        I_img = Image.open(I_path).convert('RGB')
        image_name = I_path.split('/')[-1]
        
        real_I = self.transform(I_img)
        return real_I, image_name

    def __len__(self):
        return self.I_size

def dehaze_function(hazy_image_dir, path_save_images, device="cuda", gamma_correction=True):
    """
    Dehaze images using the SLP model and save the results.

    Args:
        hazy_image_dir (str): Directory containing the hazy images.
        path_save_images (str): Directory to save the dehazed images.
        device (str): Device to run the model on ("cuda" or "cpu").
        gamma_correction (bool): Whether to apply gamma correction.
    """
    # Ensure the save directory exists
    os.makedirs(path_save_images, exist_ok=True)

    # Define data transformations and dataset loader
    data_transform = transforms.Compose([transforms.ToTensor()])
    imgLoader = torch.utils.data.DataLoader(MyDataSet_single(hazy_image_dir, data_transform), batch_size=1, shuffle=False)

    # Load the SLP model
    SLP_model = restore_SLP()
    SLP_model.to(device)
    SLP_model.eval()

    with torch.no_grad():
        for i, (haze, haze_name) in enumerate(imgLoader, 0):
            # Process the image
            dehaze_image = SLP_model(haze.to(device))

            # Apply optional gamma correction
            if gamma_correction:
                dehaze_image = dehaze_image ** 0.8

            # Save the dehazed image
            vutils.save_image(dehaze_image[0, :].to("cpu"), os.path.join(path_save_images, haze_name[0]))
            print(f"Processed and saved image {i}: {haze_name[0]}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Dehaze images using the SLP model.")
    parser.add_argument("--input_dir", type=str, default="./hazy images/", help="Directory containing the hazy images.")
    parser.add_argument("--output_dir", type=str, default="./dehazed_images/", help="Directory to save the dehazed images.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on.")
    parser.add_argument("--gamma", action="store_true", help="Apply gamma correction to the output images.")

    # Parse arguments
    args = parser.parse_args()

    # Call the dehazing function with the provided arguments
    dehaze_function(
        hazy_image_dir=args.input_dir,
        path_save_images=args.output_dir,
        device=args.device,
        gamma_correction=args.gamma
    )