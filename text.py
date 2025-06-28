import torch
from tqdm import tqdm

total_timesteps = 100
sampling_timesteps = 10
batch = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
print("times:", times)
times = list(reversed(times.int().tolist()))
print("times:", times)  
time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
print("time_pairs:", time_pairs)

shape = (batch, 3, 64, 64)  # Example shape for an image tensor
img = torch.randn(shape, device=device)
print("img shape:", img.shape)
imgs = [img]
print(("imgs length", len(imgs)))
#print("imgs:", imgs)

draft_tokens = []  # Initialize draft_tokens with the initial image

x_start = None

for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            print("time_cond:", time_cond)  #time_cond: tensor([49], device='cuda:0')

            #pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True)
            pred_noise = torch.randn_like(img)
            print("pred_noise shape:", pred_noise.shape)
            #print("pred_noise:", pred_noise)
            x_start = torch.randn_like(img)
            print("x_start shape:", x_start.shape)
            #print("x_start:", x_start)

            print(("imgs length", len(imgs)))

            if time_next < 0:
                img = x_start
                imgs.append(img)
                draft_tokens.append(img)  # Append the new image to draft_tokens
                continue

            img = x_start + \
                 pred_noise
            
            imgs.append(img)
            draft_tokens.append(img)  # Append the new image to draft_tokens

        
print(("imgs length", len(imgs)))
print("dt length:", len(draft_tokens))



@torch.no_grad()
def ddim_sample(self, img, shape, x_cond, task_embed, target_timestep, return_all_timesteps=False):
    batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #img = torch.randn(shape, device=device)
    img = img
    imgs = [img]
    draft_tokens = [img]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'): # [(99, 89), (89, 79), (79, 69), (69, 59), (59, 49), (49, 39), (39, 29), (29, 19), (19, 9), (9, -1)]
        time_cond = torch.full((batch,), time, device = device, dtype = torch.long) # 99 89 79....9
        
        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True)

        pred_noise = torch.randn_like(img)
        print("pred_noise shape:", pred_noise.shape)
        x_start = torch.randn_like(img)
        print("x_start shape:", x_start.shape)

        if time_next < 0:
            img = x_start
            imgs.append(img)
            draft_tokens.append(img)
            continue #直接结束for循环

        noise = torch.randn_like(img)
        # img就是预测的x_t-1，x_start是预测的x_0
        img = x_start + \
             pred_noise + \
             noise

        imgs.append(img)
        draft_tokens.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    #采样10步时，draft_tokens应为包含11个tensor的list[img, x99, x89, ..., x9(x_start)]
    return ret, draft_tokens

@torch.no_grad()
def target_sample(self, draft_tokens, shape, x_cond, task_embed, target_timestep, return_all_timesteps=False):
    batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #img = torch.randn(shape, device=device)
    img = draft_tokens[0]
    imgs = [img]
    target_tokens = []

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'): # [(99, 89), (89, 79), (79, 69), (69, 59), (59, 49), (49, 39), (39, 29), (29, 19), (19, 9), (9, -1)]
        time_cond = torch.full((batch,), time, device = device, dtype = torch.long) # 99 89 79....9

        pred_noise, x_start, *_ = self.model_predictions(draft_tokens, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True)
        # pred_noise和x_start都应该是一个list，包含10个tensor
        
        pred_noise = torch.randn_like(img)
        print("pred_noise shape:", pred_noise.shape)
        x_start = torch.randn_like(img)
        print("x_start shape:", x_start.shape)
        
        if time_next < 0:
            img = x_start
            imgs.append(img)
            draft_tokens.append(img)   #要改为对应位置加上img这个list
            continue #直接结束for循环

    

        noise = torch.randn_like(img)
        # img就是预测的x_t-1，x_start是预测的x_0
        img = x_start + \
            pred_noise + \
            noise

        imgs.append(img)
        draft_tokens.append(img) #要改为对应位置加上img这个list

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    return ret, draft_tokens

@torch.no_grad()
def speculative_decoding(self, shape, x_cond, text, draft_model, target_timestep, return_all_timesteps=False):
    device = self.betas.device
    img = torch.randn(shape, device=device) #初始化高斯噪声
    imgs = []
    draft_tokens = []
    target_tokens = []
    while True:
        #第一步可以加上T相关的参数target_timestpe方便后面代码
        ret, draft_tokens = self.ddim_sample(img, shape, x_cond, text, target_timestep, return_all_timesteps=True) # draft_tokens: list of torch.Tensor, each shape [B, C, H, W]
        
        #draft_tokens = [[token] for token in draft_tokens] #draft_token_batches = [[img], [x99], [x89], ..., [x9]]

        self.target_sample(draft_tokens, shape, x_cond, text, target_timestep, return_all_timesteps=True)
    
    return 

@torch.no_grad()   #sample-speculative_decoding-sample_fn
def sample(self, x_cond, task_embed, draft_model, target_timestep, batch_size = 16, return_all_timesteps = False):
    image_size, channels = self.image_size, self.channels
    sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
    ret, draft_tokens = sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed, return_all_timesteps = return_all_timesteps)
    tokens = self.speculative_decoding((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed, draft_model, target_timestep, return_all_timesteps = return_all_timesteps)
    return ret