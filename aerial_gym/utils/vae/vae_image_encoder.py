import torch
import os
from aerial_gym.utils.vae.VAE import VAE
from aerial_gym.utils.vae.AE import AE

def clean_state_dict(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims


class AEImageEncoder:
    """
    Wraps around the AE class for efficient inference for the aerial_gym class.
    替换 VAE 为确定性的 AE 模型。
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        # 1. 初始化 AE 模型而不是 VAE
        self.ae_model = AE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        
        # 组合模型路径
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        
        # 加载权重
        print("Loading AE weights from file: ", weight_file_path)
        # 使用 clean_state_dict 处理潜在的键名不匹配问题
        state_dict = clean_state_dict(torch.load(weight_file_path, map_location=device))
        self.ae_model.load_state_dict(state_dict)
        
        # 设置为评估模式
        self.ae_model.eval()

    def encode(self, image_tensors):
        """
        Encode the set of images to a latent space.
        由于 AE 是确定性的，直接返回潜在向量 z。
        """
        with torch.no_grad():
            # 保持与 VAEImageEncoder 一致的维度调整逻辑
            # squeeze 0th dimension and unsqueeze 1st dimension
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            
            # 图像插值/缩放处理
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            
            # 2. 调用 AE 的 encode
            # AE.py 中的 encode 返回 (z, z, None)，我们只需要第一个 z
            z, _, _ = self.ae_model.encode(interpolated_image)
            
        # AE 不需要区分 return_sampled_latent，因为 z 和 means 是一样的
        return z

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.ae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims