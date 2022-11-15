# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Renmin University of China (Shuai Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""XiaoiceSing related modules."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_tts_fastspeech import (
    FeedForwardTransformerLoss as XiaoiceSingLoss,  # NOQA
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding, ScaledPositionalEncoding

from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.svs.abs_svs import AbsSVS
from espnet2.tts.gst.style_encoder import StyleEncoder

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

import random
from torch.distributions import Beta

Beta_distribution = Beta(
    torch.tensor([0.5]), torch.tensor([0.5])
)  # NOTE(Shuai) Fix Me! Add to args

class PhonemeSegmentationPredictor(torch.nn.Module):
    def __init__(
        self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):

        super(PhonemeSegmentationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 4)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)
        """
        "if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()  # avoid negative value
        """
        #xs = xs.exp()

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):

        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):

        return self._forward(xs, x_masks, True)


class PhonemeLengthRegulator(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def forward(
        self, syllable, syllable_num, syllable_lengths, beat_syb, ds_alf, label_xml, label_xml_lengths, is_inference=False
    ):  # syllable, beat_syb, ds_alf, label_xml, label_xml_lengths

        #print(ds_alf)
        beat = torch.zeros(label_xml.shape).to(syllable.device)
        for i in range(len(syllable_lengths)):
            index = 0
            for j in range(syllable_lengths[i]):
                percent = ds_alf[i][j][: syllable_num[i][j]]
                #print("PER", percent)
                percent = torch.softmax(percent, dim = 0)
                #print(percent)
                #p = 0
                for k in range(syllable_num[i][j]):
                    beat[i][index] = beat_syb[i][j] * percent[k]
                    if is_inference:
                        beat[i][index] = torch.clamp(torch.round(beat[i][index]), min = 1)
                    #p += percent[k]
                    index += 1
                #beat[i][index] = beat_syb[i][j] * (1 - p)
                #index += 1
            assert index == label_xml_lengths[i]
        beat_lengths = label_xml_lengths
                
        return beat, beat_lengths  # label, midi, beat, ds_alf, len_alf

class XiaoiceSingALF(AbsSVS):
    """XiaoiceSing module for Singing Voice Synthesis.
    This is a module of XiaoiceSing. A high-quality singing voice synthesis system which
    employs an integrated network for spectrum, F0 and duration modeling. It follows the
    main architecture of FastSpeech while proposing some singing-specific design:
        1) Add features from musical score (e.g.note pitch and length)
        2) Add a residual connection in F0 prediction to attenuate off-key issues
        3) The duration of all the phonemes in a musical note is accumulated to calculate
        the syllable duration loss for rhythm enhancement (syllable loss)
    .. _`XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System`:
        https://arxiv.org/pdf/2006.06261.pdf
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        tempo_dim: int,
        syb_dim: int,
        odim: int,
        embed_dim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = "L1",
    ):
        """Initialize XiaoiceSing module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.tempo_dim = tempo_dim
        self.odim = odim
        self.embed_dim = embed_dim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define encoder
        self.syllable_input_layer = torch.nn.Embedding(
            num_embeddings=syb_dim, embedding_dim=embed_dim, padding_idx=self.padding_idx
        )
        self.midi_syb_input_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        self.beat_syb_input_layer = torch.nn.Embedding(
            num_embeddings=tempo_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )

        self.phone_encode_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=embed_dim, padding_idx=self.padding_idx
        )
        self.midi_encode_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        self.tempo_encode_layer = torch.nn.Embedding(
            num_embeddings=tempo_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
            self.syb_encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
            self.midi_syb_encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
            self.tempo_syb_encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.phoneme_length_regulator = PhonemeLengthRegulator()
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        #dim_direction = 2 if ebidirectional is True else 1
        self.phoneme_segmentation_predictor = PhonemeSegmentationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )
        self.length_regulator = LengthRegulator()
        self.seg_loss = torch.nn.MSELoss(reduction="mean")

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        self.criterion = XiaoiceSingLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        ds: torch.Tensor,
        label_lab: Optional[torch.Tensor] = None,
        label_lab_lengths: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        label_xml_lengths: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_lab_lengths: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        midi_xml_lengths: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_lab_lengths: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        tempo_xml_lengths: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_lab_lengths: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
        beat_xml_lengths: Optional[torch.Tensor] = None,
        # ALF
        syllable: Optional[torch.Tensor] = None,
        syllable_num: Optional[torch.Tensor] = None,
        syllable_lengths: Optional[torch.Tensor] = None,
        midi_syb: Optional[torch.Tensor] = None,
        midi_syb_lengths: Optional[torch.Tensor] = None,
        beat_syb: Optional[torch.Tensor] = None,
        beat_syb_lengths: Optional[torch.Tensor] = None,
        beat_note: Optional[torch.Tensor] = None,
        ds_syb: Optional[torch.Tensor] = None,
        ds_syb_lengths: Optional[torch.Tensor] = None,
        ds_xml: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.
        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """

        syllable = syllable[:, : syllable_lengths.max()]
        midi_syb = midi_syb[:, : midi_syb_lengths.max()]
        beat_syb = beat_syb[:, : beat_syb_lengths.max()]

        syllable_emb = self.syllable_input_layer(syllable)  # B, len, dim
        midi_syb_emb = self.midi_syb_input_layer(midi_syb)
        beat_syb_emb = self.beat_syb_input_layer(beat_syb)
        
        x_masks = self._source_mask(syllable_lengths)

        hs_syllable, _ = self.syb_encoder(syllable_emb, x_masks)
        hs_midi_syb, _ = self.midi_syb_encoder(midi_syb_emb, x_masks)
        hs_beat_syb, _ = self.tempo_syb_encoder(beat_syb_emb, x_masks)

        h_ds = hs_syllable + hs_midi_syb + hs_beat_syb
        #h_ds, _ = self.syb_encoder(syb_input_emb, x_masks)

        ds_alf = self.phoneme_segmentation_predictor(h_ds)

        beat, beat_lengths = self.phoneme_length_regulator(
            syllable, syllable_num, syllable_lengths, beat_syb, ds_alf, label_xml, label_xml_lengths
        )

        tempo = beat_lab
        seg_loss = self.seg_loss(beat.to(dtype=torch.float), tempo.to(dtype=torch.float)) * 0.02

        # end ALF

        label = label_xml
        midi = midi_xml
        #tempo = beat_xml
        label_lengths = label_xml_lengths
        midi_lengths = midi_xml_lengths
        tempo_lengths = beat_xml_lengths

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        tempo_emb = self.tempo_encode_layer(tempo)
        input_emb = label_emb + midi_emb + tempo_emb

        x_masks = self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor(hs, d_masks)  # (B, T_text)

        hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)

        # forward decoder
        h_masks = self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            assert feats_lengths.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = feats_lengths.new(
                [olen - olen % self.reduction_factor for olen in feats_lengths]
            )
            max_olen = max(olens)
            ys = feats[:, :max_olen]
        else:
            ys = feats
            olens = feats_lengths

        ilens = label_lengths
        l1_loss, duration_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens
        )
        loss = l1_loss + duration_loss + seg_loss

        stats = dict(
            loss=loss.item(),
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            seg_loss=seg_loss.item(),
        )

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if flag_IsValid == False:
            return loss, stats, weight
        else:
            return loss, stats, weight, after_outs[:, : olens.max()], ys, olens

    def inference(
        self,
        text: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        label_lab: Optional[torch.Tensor] = None,
        label_xml: Optional[torch.Tensor] = None,
        midi_lab: Optional[torch.Tensor] = None,
        midi_xml: Optional[torch.Tensor] = None,
        tempo_lab: Optional[torch.Tensor] = None,
        tempo_xml: Optional[torch.Tensor] = None,
        beat_lab: Optional[torch.Tensor] = None,
        beat_xml: Optional[torch.Tensor] = None,
        syllable: Optional[torch.Tensor] = None,
        syllable_num: Optional[torch.Tensor] = None,
        syllable_lengths: Optional[torch.Tensor] = None,
        label_xml_lengths: Optional[torch.Tensor] = None,
        midi_syb: Optional[torch.Tensor] = None,
        beat_syb: Optional[torch.Tensor] = None,
        beat_note: Optional[torch.Tensor] = None,
        ds_syb: Optional[torch.Tensor] = None,
        ds_xml: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        # alpha: float = 1.0,
        # use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.
        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            durations (Optional[LongTensor]): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
        """

        #syllable = syllable[:, : syllable_lengths.max()]
        #midi_syb = midi_syb[:, : midi_syb_lengths.max()]
        #beat_syb = beat_syb[:, : beat_syb_lengths.max()]

        syllable_emb = self.syllable_input_layer(syllable)  # B, len, dim
        midi_syb_emb = self.midi_syb_input_layer(midi_syb)
        beat_syb_emb = self.beat_syb_input_layer(beat_syb)

        x_masks = self._source_mask(syllable_lengths)

        hs_syllable, _ = self.syb_encoder(syllable_emb, x_masks)
        hs_midi_syb, _ = self.midi_syb_encoder(midi_syb_emb, x_masks)
        hs_beat_syb, _ = self.tempo_syb_encoder(beat_syb_emb, x_masks)

        h_ds = hs_syllable + hs_midi_syb + hs_beat_syb
        #h_ds, _ = self.syb_encoder(syb_input_emb, x_masks)

        ds_alf = self.phoneme_segmentation_predictor(h_ds)

        beat, beat_lengths = self.phoneme_length_regulator(
            syllable, syllable_num, syllable_lengths, beat_syb, ds_alf, label_xml, label_xml_lengths, is_inference=True
        )

        tempo = torch.round(beat).to(torch.int64)
        #seg_loss = self.seg_loss(beat.to(dtype=torch.float), tempo.to(dtype=torch.float)) * 0.02

        # end ALF

        label = label_xml
        midi = midi_xml
        #tempo = beat_xml
        #label_lengths = label_xml_lengths
        #midi_lengths = midi_xml_lengths
        #tempo_lengths = beat_xml_lengths

        #label = label_xml
        #midi = midi_xml
        #tempo = beat_xml

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        tempo_emb = self.tempo_encode_layer(tempo)
        input_emb = label_emb + midi_emb + tempo_emb

        x_masks = None  # self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # forward duration predictor and length regulator
        d_masks = None  # make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)

        logging.info(f"ds: {ds}")
        logging.info(f"ds.shape: {ds.shape}")
        logging.info(f"d_outs: {d_outs}")
        logging.info(f"d_outs.shape: {d_outs.shape}")

        # use G.T. duration
        # hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)

        # use duration model output
        hs = self.length_regulator(hs, d_outs_int)  # (B, T_feats, adim)

        # forward decoder
        h_masks = None  # self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # ilens = torch.tensor([len(label)])
        # olens = torch.tensor([len(feats)])
        # ys = feats
        # l1_loss, duration_loss = self.criterion(
        #     after_outs, before_outs, d_outs, ys, ds, ilens, olens
        # )
        # loss = l1_loss + duration_loss
        # logging.info(f"loss: {loss}, l1_loss: {l1_loss}, duration_loss: {duration_loss}")

        return after_outs, None, None

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Args:
            ilens (LongTensor): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

