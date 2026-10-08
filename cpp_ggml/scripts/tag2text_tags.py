"""Run the official Tag2Text tagging branch without text generation."""

import torch


def tag_logits(model, image):
    image_embeds = model.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)
    label_embed = model.label_embed.weight.unsqueeze(0).expand(image.size(0), -1, -1)
    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode="tagging",
    )
    return model.fc(tagging_embed[0])
