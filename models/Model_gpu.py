import torch
import torch.nn.functional as F
from .layers_gpu import *
from .Attention_fusion import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class BDHNet(nn.Module):
    def __init__(self, num_res=20, base_channel=32, event_channels=12, 
                 num_heads=None, ffn_expansion_factor=4, pretrained_path=None):
        """
        Args:
            num_res: Number of residual blocks in each EBlock/DBlock
            base_channel: Base channel number for feature extraction (default: 32)
            event_channels: Number of event channels/bin (default: 12)
            num_heads: Number of attention heads for RBAM. Can be int or list/tuple of 3 ints for [max, mid, min].
                       Default: [2, 4, 4] for base_channel=32
            ffn_expansion_factor: Expansion factor for FFN in RBAM (default: 4)
            pretrained_path: Path to pretrained weights
        """
        super(BDHNet, self).__init__()
        # Handle num_heads parameter
        if num_heads is None:
            num_heads = [2, 4, 4]  # Default heads for [max, mid, min] scales
        elif isinstance(num_heads, int):
            num_heads = [num_heads, num_heads, num_heads]
        elif len(num_heads) != 3:
            raise ValueError("num_heads must be an int or a list/tuple of 3 integers")
        
        # Store configuration for external access
        self.base_channel = base_channel
        self.event_channels = event_channels
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

        # Load pretrained model if specified
        if pretrained_path:
            print(pretrained_path)
            self.load_pretrained(pretrained_path)

        self.Encoder_eframe = nn.ModuleList([
            NCM(base_channel, base_channel, kernel_size=3, stride=1, lif=True, norm=True),
            NCM(base_channel*2, base_channel*2, kernel_size=3, stride=1, lif=True, norm=True),
            NCM(base_channel*4, base_channel*4, kernel_size=3, stride=1, lif=True, norm=True),
        ])

        self.feat_extract_eframe = nn.ModuleList([
            NCM(1, base_channel, kernel_size=3, stride=1),
            NCM(base_channel, base_channel*2, kernel_size=3, stride=2),
            NCM(base_channel*2, base_channel*4, kernel_size=3, stride=2)
        ])

        self.s2f_max = nn.Conv2d(event_channels, 1, 1, padding=0, stride=1)
        self.s2f_mid = nn.Conv2d(event_channels, 1, 1, padding=0, stride=1)
        self.s2f_min = nn.Conv2d(event_channels, 1, 1, padding=0, stride=1)

        self.event_init = nn.Conv2d(event_channels, 1, 1, padding=0, stride=1)

        self.fusion_max = RBAM(base_channel, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')
        self.fusion_mid = RBAM(base_channel*2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')
        self.fusion_min = RBAM(base_channel*4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias')

    def load_pretrained(self, pretrained_path):
        full_dict = torch.load(pretrained_path, map_location=torch.device('cuda'))
        pretrained_dict = full_dict['model'] if 'model' in full_dict else full_dict
        self.load_state_dict(pretrained_dict)

    def _forward_single(self, x_blur, event_frame):
        ##frame
        x_2 = F.interpolate(x_blur, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        event_init_max = self.event_init(event_frame)
        event_init_mid = F.interpolate(event_init_max, scale_factor=0.5)
        event_init_min = F.interpolate(event_init_mid, scale_factor=0.5)

        outputs = list()

        x_ = self.feat_extract[0](x_blur)
        res1_frame = self.Encoder[0](x_)

        v_init_max = res1_frame + event_init_max
        x_eframe, v_max = self.feat_extract_eframe[0](event_frame, v_init_max)
        res1_event_ori, _ = self.Encoder_eframe[0](x_eframe, v_max)

        res1_event_ori_ = res1_event_ori + x_eframe

        res1_event = self.s2f_max(res1_event_ori_)

        res1_event = res1_event.permute(1, 0, 2, 3)

        spike_out = res1_event

        res1, mask1 = self.fusion_max(res1_frame, res1_event, res1_event_ori_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2_frame = self.Encoder[1](z)

        v_init_mid = res2_frame + event_init_mid
        x_eframe_1, v_mid = self.feat_extract_eframe[1](res1_event_ori_, v_init_mid)
        res1_event_1_ori, _ = self.Encoder_eframe[1](x_eframe_1, v_mid)

        res1_event_1_ori_ = res1_event_1_ori + x_eframe_1

        res1_event_1 = self.s2f_mid(res1_event_1_ori_)
        res1_event_1 = res1_event_1.permute(1, 0, 2, 3)

        res2, mask2 = self.fusion_mid(res2_frame, res1_event_1, res1_event_1_ori_)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        v_init_min = z + event_init_min
        x_eframe_2, v_min = self.feat_extract_eframe[2](res1_event_1_ori_, v_init_min)
        res1_event_2_ori, _ = self.Encoder_eframe[2](x_eframe_2, v_min)

        res1_event_2_ori_ = res1_event_2_ori + x_eframe_2

        res1_event_2 = self.s2f_min(res1_event_2_ori_)
        res1_event_2 = res1_event_2.permute(1, 0, 2, 3)

        z, mask3 = self.fusion_min(z, res1_event_2, res1_event_2_ori_)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x_blur)

        return outputs, spike_out, mask1

    def forward(self, x_blur, event_frame):

        batch_size = x_blur.shape[0]
        
        if batch_size == 1:
            # batch_size=1 时，直接使用原始逻辑
            return self._forward_single(x_blur, event_frame)
        else:
            # batch_size>1 时，逐个样本处理
            outputs_list = [[] for _ in range(3)]  # 3 个尺度的输出
            spike_out_list = []
            mask1_list = []
            
            for i in range(batch_size):
                # 提取单个样本
                x_blur_i = x_blur[i:i+1]
                event_frame_i = event_frame[i:i+1]
                
                # 处理单个样本
                outputs_i, spike_out_i, mask1_i = self._forward_single(x_blur_i, event_frame_i)
                
                # 收集结果
                for j, out in enumerate(outputs_i):
                    outputs_list[j].append(out)
                spike_out_list.append(spike_out_i)
                mask1_list.append(mask1_i)
            
            # 合并结果
            outputs = [torch.cat(outputs_list[j], dim=0) for j in range(3)]
            spike_out = torch.cat(spike_out_list, dim=0)
            mask1 = torch.cat(mask1_list, dim=0)
            
            return outputs, spike_out, mask1
