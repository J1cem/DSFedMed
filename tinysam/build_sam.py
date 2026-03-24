# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT, MultiSam, DatasetClassifier, GatingNet, MoeSam, MoeDecoder, DTEncoder 


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args,checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_sam_decoder_moe(args,checkpoint=None):
    return _build_sam_decoder_moe(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_sam_vit_b_multi(args,checkpoint=None):
    return _build_sam_multi(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )  


def build_sam_vit_b_moe(args, checkpoint=None):
    return _build_sam_moe(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        num_decoders=3,  # **新增参数，指定解码器数量**
        checkpoint=checkpoint,
    )


def build_sam_vit_t(args, checkpoint=None):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes= args.num_classes
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            # print(state_dict)
        mobile_sam.load_state_dict(state_dict, strict = False)
        print('load sucessfully',checkpoint,'\n')
    return mobile_sam

def build_sam_vit_g(args, checkpoint=None):
    prompt_embed_dim = 256
    image_size = args.image_size
    domain_num = args.client_num
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    general_sam = Sam(
            image_encoder=DTEncoder(
                depth=12,
                embed_dim=768,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=256,
                domain_num=domain_num
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes= args.num_classes
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    general_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            # print(state_dict)
        general_sam.load_state_dict(state_dict, strict = False)
        print('load general sam sucessfully',checkpoint,'\n')
    return general_sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
    "multiple": build_sam_vit_b_multi,
    "moe": build_sam_vit_b_moe,
    "decoder_moe": build_sam_decoder_moe,
    "general": build_sam_vit_g
}


def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes= args.num_classes
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            # print(state_dict['state_dict'])
        if 'state_dict' in state_dict.keys():
            sam.load_state_dict(state_dict['state_dict'], strict = False)
        else:
            sam.load_state_dict(state_dict, strict = False)
        print('load sucessfully',checkpoint,'\n')
    return sam

def _build_sam_decoder_moe(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    m_decoder_sam = Sam(
            image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MoeDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes= args.num_classes
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    m_decoder_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            # print(state_dict)
        m_decoder_sam.load_state_dict(state_dict, strict = False)
        print('load sucessfully',checkpoint,'\n')
    return m_decoder_sam


def _build_sam_moe(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_decoders,  # **新增：可变数量的解码器**
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # 图像编码器
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # 提示编码器
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    # 数据集分类器 (MLP)
    dataset_classifier = GatingNet(embedding_dim=256, hidden_dim=128, num_class=num_decoders)

    # 第一个 mask decoder
    mask_decoder_1 = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=args.num_classes,
    )

    # **新增** 第二个 mask decoder
    mask_decoder_2 = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=args.num_classes,
    )

    # **新增** 第三个 mask decoder
    mask_decoder_3 = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=args.num_classes,
    )

    # 组装 Sam
    sam = MoeSam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        dataset_classifier=dataset_classifier,  # **新增**
        mask_decoder_1=mask_decoder_1,  # **修改**
        mask_decoder_2=mask_decoder_2,  # **新增**
        mask_decoder_3=mask_decoder_3,  # **新增**
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    sam.eval()

    # 载入权重
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']  # 提取 state_dict

        # 复制 mask_decoder 的权重到 mask_decoder_1 和 mask_decoder_2
        updated_state_dict = state_dict.copy()  # 复制 state_dict 避免修改原数据
        for key in state_dict.keys():
            if "mask_decoder" in key:
                key1 = key.replace("mask_decoder", "mask_decoder_1")
                key2 = key.replace("mask_decoder", "mask_decoder_2")
                key3 = key.replace("mask_decoder", "mask_decoder_3")
                updated_state_dict[key1] = state_dict[key]
                updated_state_dict[key2] = state_dict[key]
                updated_state_dict[key3] = state_dict[key]

        # 载入权重并检查未匹配的参数
        missing_keys, unexpected_keys = sam.load_state_dict(updated_state_dict, strict=False)

        print("Model weights loaded from:", checkpoint)
        if missing_keys:
            print("Missing keys (not found in checkpoint):", missing_keys)


    return sam


def _build_sam_multi(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # 图像编码器
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # 提示编码器
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    # **新增** 数据集分类器 (MLP)
    dataset_classifier = DatasetClassifier(embedding_dim=256, hidden_dim=128)

    # 第一个 mask decoder
    mask_decoder_1 = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=args.num_classes,
    )

    # **新增** 第二个 mask decoder
    mask_decoder_2 = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=args.num_classes,
    )

    # 组装 Sam
    sam = MultiSam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        dataset_classifier=dataset_classifier,  # **新增**
        mask_decoder_1=mask_decoder_1,  # **修改**
        mask_decoder_2=mask_decoder_2,  # **新增**
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    sam.eval()

    # 载入权重
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']  # 提取 state_dict

        # 复制 mask_decoder 的权重到 mask_decoder_1 和 mask_decoder_2
        updated_state_dict = state_dict.copy()  # 复制 state_dict 避免修改原数据
        for key in state_dict.keys():
            if "mask_decoder" in key:
                key1 = key.replace("mask_decoder", "mask_decoder_1")
                key2 = key.replace("mask_decoder", "mask_decoder_2")
                updated_state_dict[key1] = state_dict[key]
                updated_state_dict[key2] = state_dict[key]

        # 载入权重并检查未匹配的参数
        missing_keys, unexpected_keys = sam.load_state_dict(updated_state_dict, strict=False)

        print("Model weights loaded from:", checkpoint)
        if missing_keys:
            print("Missing keys (not found in checkpoint):", missing_keys)


    return sam