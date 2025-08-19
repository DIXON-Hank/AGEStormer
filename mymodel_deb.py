import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import to_2tuple,trunc_normal_
from torchinfo import summary

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) Regularization"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x  # No drop in evaluation mode or if drop_prob=0

        # Generate a random mask at the batch level
        keep_prob = 1 - self.drop_prob
        random_tensor = torch.rand(x.shape, device=x.device, dtype=x.dtype) < keep_prob
        output = x * random_tensor
        return output

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def _generate_positional_encoding(self, H, W, d_model):
        """
        Generate sine-cosine positional encoding dynamically based on H, W.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = torch.zeros((H * W, d_model), device=device)
        y_pos = torch.arange(0, H, device=device).repeat_interleave(W)
        x_pos = torch.arange(0, W, device=device).repeat(H)

        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)

        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Project image to patches
        x = self.proj(x)
        _, _, H, W = x.shape

        # Flatten and transpose
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        # Generate positional encoding dynamically
        positional_encoding = self._generate_positional_encoding(H, W, self.embed_dim).to(x.device)

        # Add positional encoding
        x = x + positional_encoding

        return x, H, W

# class DynamicSparseAttention(nn.Module):
#     """Dynamic Sparse Attention"""

#     def __init__(self, dim, num_heads, top_k=8, attn_drop=0., proj_drop=0.):
#         super(DynamicSparseAttention, self).__init__()
#         self.num_heads = num_heads
#         self.top_k = top_k
#         self.scale = (dim // num_heads) ** -0.5

#         # Linear projections for Q, K, V
#         self.q = nn.Linear(dim, dim)
#         self.kv = nn.Linear(dim, dim * 2)
#         self.proj = nn.Linear(dim, dim)
#         self.activation = nn.ReLU()

#         # Dropout layers
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self._init_weights()

#     def forward(self, x):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         q = self.activation(q)
#         kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         # Compute attention scores
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         topk_indices = attn.topk(self.top_k, dim=-1, sorted=False)[1]  # Select top-k
#         sparse_mask = torch.zeros_like(attn).scatter_(-1, topk_indices, 1)
#         sparse_attn = attn * sparse_mask
#         sparse_attn = self.attn_drop(torch.softmax(sparse_attn, dim=-1))
#         # Compute output
#         out = (sparse_attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#         return out

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

class MSFN(nn.Module):
    """Mixed-Scale Feed-forward Network"""

    def __init__(self, dim, hidden_dim, drop=0.):
        super(MSFN, self).__init__()
        self.conv3 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=dim)
        self.conv5 = nn.Conv2d(dim, hidden_dim, kernel_size=5, padding=2, groups=dim)
        self.conv1x1 = nn.Conv2d(2 * hidden_dim, dim, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv1x1.weight)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)
        if self.conv5.bias is not None:
            nn.init.constant_(self.conv5.bias, 0)
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        assert H * W ==N, f"H * W ({H} * {W}) must equal N ({N})"

        x = x.transpose(1, 2).view(B, C, H, W)

        # Multi-scale convolutions
        x3 = self.activation(self.conv3(x))
        x5 = self.activation(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)  # Concatenate multi-scale features
        x = self.conv1x1(x)  # Reduce to original dimension

        x = x.flatten(2).transpose(1, 2)  # Restore back to shape (B, N, C)
        x = self.dropout(x)
        return x

# class STB(nn.Module):
#     def __init__(self, dim, num_heads=8, top_k=8, mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.):
#         super(STB, self).__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = DynamicSparseAttention(dim, num_heads, top_k, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

#     def forward(self, x, H, W): 
#         # Apply attention block with residual connection
#         x = x + self.drop_path1(self.attn(self.norm1(x)))

#         # Apply MLP block with residual connection
#         x = x + self.drop_path2(self.mlp(self.norm2(x), H, W))
#         return x

class DetailEnhancedBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., sr_ratio=1, detail_kernel_size=5):
        super().__init__()
        # 正常 Transformer 组件
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias)
        
        # 细节注意力模块
        self.detail_conv = nn.Conv2d(dim, dim, kernel_size=detail_kernel_size, padding=detail_kernel_size // 2, groups=dim)
        self.detail_norm = nn.LayerNorm(dim)
        self.detail_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()  # 输出动态权重
        )
        
        # DropPath 用于随机丢弃
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MSFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        # (1) 自注意力机制
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        
        # (2) 提取细节特征
        x_reshaped = x.permute(0, 2, 1).reshape(B, C, H, W)  # 还原到图像形状
        detail_features = self.detail_conv(x_reshaped)  # 提取细节特征
        detail_features = detail_features.flatten(2).transpose(1, 2)  # 恢复为 Transformer 格式
        
        # (3) 计算细节权重
        detail_weights = self.detail_mlp(self.detail_norm(detail_features))  # 细节权重
        
        # (4) 细节增强
        x = x + detail_weights * detail_features  # 将细节权重作用到全局特征
        
        # (5) FFN
        x = x + self.drop_path(self.mlp(self.norm2(x),H,W))
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        pool = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)
        gate = self.sigmoid(self.conv(pool))  # (B, 1, H, W)
        return x * gate  # 对输入特征按空间维度加权

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
    
class ConvLayer(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_chans, out_chans, kernel_size, stride=stride, padding=1)
        # self.smooth = nn.Conv2d(out_chans, out_chans, 3, 1, 1)

    def forward(self, x):
        out = self.conv2d(x)
        # out = self.smooth(out)
        return out

class MultiScaleFusion(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Skip connection from encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    
class OutputLayer(nn.Module):
    def __init__(self, in_chans, out_chans=3):
        super(OutputLayer,self).__init__()

        self.upsample2 = UpsampleConvLayer(in_chans, in_chans // 2, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(in_chans // 2))
        # self.upsample1 = UpsampleConvLayer(in_chans // 2, in_chans // 4, kernel_size=4, stride=2)
        # self.dense_1 = nn.Sequential(ResidualBlock(in_chans // 4))

        # self.conv_output = ConvLayer(in_chans // 4, out_chans, kernel_size=3, stride=1, padding=1)
        self.conv_output = ConvLayer(in_chans // 2, out_chans, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()

    def forward(self, x):
        x = self.upsample2(x)
        x = self.dense_2(x)
        # x = self.upsample1(x)
        # x = self.dense_1(x)
        return self.active(self.conv_output(x))

class AGRestormer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 256, 512], num_heads=[8, 8, 4, 4],
                 mlp_ratio=4.0, blocks_num=[4, 4, 2, 2], drop_path_rate=0.1, refine_blocks=3):
        super().__init__()
        # get drop_path_rate for each layer
        self.embed_dims = embed_dims
        depth = sum(blocks_num)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dpr1 = dpr[0:sum(blocks_num[:1])]
        dpr2 = dpr[sum(blocks_num[:1]):sum(blocks_num[:2])]
        dpr3 = dpr[sum(blocks_num[:2]):sum(blocks_num[:3])]
        dpr4 = dpr[sum(blocks_num[:3]):sum(blocks_num[:4])]

        self.embed_layer0 = PatchEmbed(patch_size=3, stride=2, in_chans=in_chans, embed_dim=embed_dims[0])

        self.embed_layer1 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.embed_layer2 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.embed_layer3 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.encoder1 = self._make_blk_enc(blocks_num[0], embed_dims[0], num_heads[0],mlp_ratio, drop_path=dpr1, detail_kernel_size=3)
        self.encoder2 = self._make_blk_enc(blocks_num[1], embed_dims[1], num_heads[1],mlp_ratio, drop_path=dpr2, detail_kernel_size=5)
        self.encoder3 = self._make_blk_enc(blocks_num[2], embed_dims[2], num_heads[2],mlp_ratio, drop_path=dpr3, detail_kernel_size=5)

        self.bottleneck = self._make_blk_enc(blocks_num[3], embed_dims[3], num_heads[3],mlp_ratio, drop_path=dpr4, detail_kernel_size=3)

        self.decoder3 = self._make_blk_dec(blocks_num[2], embed_dims[2], num_heads[2], mlp_ratio, detail_kernel_size=5)
        self.decoder2 = self._make_blk_dec(blocks_num[1], embed_dims[1], num_heads[1], mlp_ratio, detail_kernel_size=5)
        self.decoder1 = self._make_blk_dec(blocks_num[0], embed_dims[0], num_heads[0], mlp_ratio, detail_kernel_size=3)

        self.upsample3 = UpsampleConvLayer(embed_dims[3], embed_dims[2], kernel_size=4, stride=2)
        self.upsample2 = UpsampleConvLayer(embed_dims[2], embed_dims[1], kernel_size=4, stride=2)
        self.upsample1 = UpsampleConvLayer(embed_dims[1], embed_dims[0], kernel_size=4, stride=2)

        self.fusion3 = MultiScaleFusion(embed_dims[2] * 2, embed_dims[2])
        self.fusion2 = MultiScaleFusion(embed_dims[1] * 2, embed_dims[1])
        self.fusion1 = MultiScaleFusion(embed_dims[0] * 2, embed_dims[0])

        self.refinement = self._make_refinement(refine_blocks, embed_dims[0], num_heads=8, mlp_ratio=mlp_ratio)

        self.output_layer = OutputLayer(embed_dims[0], 3)
        
        self.attn_gate = SpatialAttention()
        

    # def _make_blk_enc(self, block_num, dim, num_heads, mlp_ratio, drop_path=0., top_k=8):
    #     """Create a layer with multiple Sparse Transformer Blocks."""
    #     return nn.Sequential(*[
    #         STB(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], top_k=top_k)
    #         for i in range(block_num)
    #     ])
    
    # def _make_blk_dec(self, block_num, dim, num_heads, mlp_ratio, top_k=8):
    #     """Create a layer with multiple Sparse Transformer Blocks."""
    #     return nn.Sequential(*[
    #         STB(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, top_k=top_k)
    #         for i in range(block_num)
    #     ])
    
    def _make_blk_enc(self, block_num, dim, num_heads, mlp_ratio, detail_kernel_size=3, drop_path=0.):
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            DetailEnhancedBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], detail_kernel_size=detail_kernel_size)
            for i in range(block_num)
        ])
    
    def _make_blk_dec(self, block_num, dim, num_heads, mlp_ratio, detail_kernel_size):
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            DetailEnhancedBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, detail_kernel_size=detail_kernel_size)
            for i in range(block_num)
        ])
    
    def _make_refinement(self, block_num, dim, num_heads, mlp_ratio):
        """Create a layer with multiple Sparse Transformer Blocks."""
        return nn.Sequential(*[
            DetailEnhancedBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(block_num)
        ])

    def forward(self, x):
        # Original input for skip connection
        enc_out=[]
        enc_out.append(x)
        
        # Patch Embedding
        x, H, W = self.embed_layer0(x)
        # ------ Encoder ------
        for blk in self.encoder1:
            x = blk(x, H, W)
        x = x.transpose(1,2).contiguous().view(-1, self.embed_layer0.embed_dim, H, W)
        enc_out.append(self.attn_gate(x))
        x, H, W = self.embed_layer1(x)

        for blk in self.encoder2:
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer1.embed_dim, H, W)
        enc_out.append(self.attn_gate(x))
        x, H, W = self.embed_layer2(x)

        for blk in self.encoder3:
            x = blk(x, H, W)        
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer2.embed_dim, H, W)
        enc_out.append(self.attn_gate(x))
        x, H, W = self.embed_layer3(x)

        for blk in self.bottleneck:
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer3.embed_dim, H, W) # B,512,7,7
        
        # ------ Decoder ------
        x = self.upsample3(x)
        H *=2
        W *=2
        f3 = torch.concat((x, enc_out[-1]), 1) # concat 
        x = self.fusion3(f3) # 1,256,14,14
        x = x.flatten(2).transpose(1, 2)
        for blk in self.decoder3:
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer2.embed_dim, H, W)
        
        x = self.upsample2(x)
        H *=2
        W *=2
        f2 = torch.concat((x, enc_out[-2]), 1) # concat 
        x = self.fusion2(f2) # 1,256,14,14
        x = x.flatten(2).transpose(1, 2)
        for blk in self.decoder2:
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer1.embed_dim, H, W)
        
        x = self.upsample1(x)
        H *=2
        W *=2
        f1 = torch.concat((x, enc_out[-3]), 1) # concat 
        x = self.fusion1(f1) # 1,256,14,14
        x = x.flatten(2).transpose(1, 2)
        for blk in self.decoder1:
            x = blk(x, H, W)
        
        # refinement
        for blk in self.refinement: 
            x = blk(x, H, W)
        x = x.transpose(1, 2).contiguous().view(-1, self.embed_layer0.embed_dim, H, W)
        
        #output layer
        x = self.output_layer(x)
        x = x + enc_out[-4] # skip connection from the original image
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AGRestormer()
model = model.to(device)
summary(model, input_size=(1, 3, 224, 224))