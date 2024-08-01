# CLIP-Surgery-Simple
This respository help you to add self-self attention without creating new model / modules, nor copying params in the forward.

***Paper***: CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks ([arxiv](https://arxiv.org/abs/2304.05653), [original implemetation](https://github.com/xmed-lab/CLIP_Surgery/tree/master/clip))

From the original vision transformer. Following are what added.

**1. Define new `forward` function of SelfAttention with dual outputs:**
```python
def new_forward(self, x):
    B, N, C = x.shape
    qkv = x @ self.in_proj_weight.t() + self.in_proj_bias.view(1, 1, -1)
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2] # B, nh, T, C
    x = F.scaled_dot_product_attention(torch.cat((v, q)),
                                        torch.cat((v, k)),
                                        torch.cat((v, v)),)
    x = x.transpose(1, 2).view(2*B, N, C)
    x = self.out_proj(x)
    x, x_ori = x.chunk(2, dim=0)
    return [x, x_ori]

```

***2. Pass the new forward function to some last attention layers in the `__init__` method of VisionTrasformer***

```python
for  i in range(1,7):
    # redefine foward method for the last 6 blocks
    self.transformer.resblocks[-i].attn.forward = new_forward.__get__(self.transformer.resblocks[-i].attn)
    # set flag for architecture surgery
    self.transformer.resblocks[-i].attn.is_surgery = True
```

***3. Update forward method of `ResidualAttentionBlock` with the flag `is_surgery`***
```python


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if getattr(self.attn, 'is_surgery', False):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x):
        # dual paths for blocks deeper than "d"
        if getattr(self.attn, 'is_surgery', False):
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x
```

***4. Update `forward` method of `VisionTransformer`***

```python

x, x_ori = self.transformer(x)
x[0, :, :] = x_ori[0, :, :] # clip_surgery
x = x.permute(1, 0, 2)  # LND -> NLD

x = self.ln_post(x)
x = x @ self.proj

return x
```

Please visit the original implemetation [here](https://github.com/xmed-lab/CLIP_Surgery/tree/master/clip)!
