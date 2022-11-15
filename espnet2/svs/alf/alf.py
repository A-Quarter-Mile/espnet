# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""NaiveRNN-DP-ALF-SVS related modules."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.svs.abs_svs import AbsSVS
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet.nets.pytorch_backend.e2e_tts_fastspeech import (
    FeedForwardTransformerLoss as FastSpeechLoss,
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.rnn.attentions import AttForward, AttForwardTA, AttLoc

class PhonemeSegmentationPredictor(torch.nn.Module):
    """Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(
        self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
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
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, True)


class PhonemeSegmentationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """Initilize duration predictor loss module.

        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(PhonemeSegmentationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss



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
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        #print(ds_alf)
        beat = torch.zeros(label_xml.shape).to(syllable.device)
        for i in range(len(syllable_lengths)):
            index = 0
            for j in range(syllable_lengths[i]):
                percent = ds_alf[i][j][: syllable_num[i][j]]
                #print("PER", percent)
                percent = torch.softmax(percent, dim = 0)
                #print(syllable[i][j], percent)
                #p = 0
                for k in range(syllable_num[i][j]):
                    beat[i][index] = beat_syb[i][j] * percent[k]
                    if is_inference:
                        beat[i][index] = torch.clamp(torch.round(beat[i][index]), min = 1)
                    #print(index, beat[i][index])
                    #p += percent[k]
                    index += 1
                #beat[i][index] = beat_syb[i][j] * (1 - p)
                #index += 1
            assert index == label_xml_lengths[i]
        beat_lengths = label_xml_lengths
                
        return beat, beat_lengths  # label, midi, beat, ds_alf, len_alf

class ALF(AbsSVS):
    """NaiveRNNDPALF-SVS module.

    This is an implementation of naive RNN with duration prediction
    for singing voice synthesis
    The features are processed directly over time-domain from music score and
    predict the singing voice features
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        tempo_dim: int,
        syb_dim: int,
        odim: int,
        embed_dim: int = 512,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        elayers: int = 3,
        eunits: int = 1024,
        ebidirectional: bool = True,
        midi_embed_integration_type: str = "add",
        dlayers: int = 3,
        dunits: int = 1024,
        dbidirectional: bool = True,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        use_batch_norm: bool = True,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        eprenet_dropout_rate: float = 0.5,
        edropout_rate: float = 0.1,
        ddropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize NaiveRNN module.

        Args: TODO(Yuning)
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.tempo_dim = tempo_dim
        self.syb_dim = syb_dim
        self.eunits = eunits
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor

        self.midi_embed_integration_type = midi_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # ALF

        self.syllable_input_layer = torch.nn.Embedding(
            num_embeddings=syb_dim, embedding_dim=eunits, padding_idx=self.padding_idx
        )
        self.midi_syb_input_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=eunits,
            padding_idx=self.padding_idx,
        )
        self.beat_syb_input_layer = torch.nn.Embedding(
            num_embeddings=tempo_dim,
            embedding_dim=eunits,
            padding_idx=self.padding_idx,
        )
        dim_direction = 2 if ebidirectional is True else 1
        self.phoneme_segmentation_predictor = PhonemeSegmentationPredictor(
            idim=eunits * dim_direction,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        self.phoneme_length_regulator = PhonemeLengthRegulator()


        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            self.encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
            self.midi_encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=midi_dim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
            self.tempo_encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=midi_dim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
        else:
            self.encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
            )
            self.midi_encoder_input_layer = torch.nn.Embedding(
                num_embeddings=midi_dim,
                embedding_dim=eunits,
                padding_idx=self.padding_idx,
            )
            self.tempo_encoder_input_layer = torch.nn.Embedding(
                num_embeddings=tempo_dim,
                embedding_dim=eunits,
                padding_idx=self.padding_idx,
            )

        self.syb_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.midi_syb_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.tempo_syb_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.seg_loss = torch.nn.MSELoss(reduction="mean")

        self.encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.midi_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.tempo_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        dim_direction = 2 if ebidirectional is True else 1
        if self.midi_embed_integration_type == "add":
            self.midi_projection = torch.nn.Linear(
                eunits * dim_direction, eunits * dim_direction
            )
        else:
            self.midi_projection = torch.nn.linear(
                3 * eunits * dim_direction, eunits * dim_direction
            )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=eunits * dim_direction,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        self.decoder = torch.nn.LSTM(
            input_size=eunits * dim_direction,
            hidden_size=dunits,
            num_layers=dlayers,
            batch_first=True,
            dropout=ddropout_rate,
            bidirectional=dbidirectional,
            # proj_size=dunits,
        )

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, dunits * dim_direction)
        self.langs = None
        if langs is not None and langs > 1:
            # TODO(Yuning): not encode yet
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, dunits * dim_direction)

        # define projection layer
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(
                    self.spk_embed_dim, dunits * dim_direction
                )
            else:
                self.projection = torch.nn.Linear(
                    dunits * dim_direction + self.spk_embed_dim, dunits * dim_direction
                )

        # define final projection
        self.feat_out = torch.nn.Linear(dunits * dim_direction, odim * reduction_factor)

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

        # define loss function
        self.criterion = FastSpeechLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
        )
        dec_idim = eunits
        adim = 512
        aconv_chans = 32
        aconv_filts = 15
        prenet_layers = 2
        prenet_units = 256
        output_activation_fn = None
        cumulate_att_w=True
        att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        self.dec = Decoder(
            idim=dec_idim,
            odim=odim,
            att=att,
            dlayers=dlayers,
            dunits=dunits,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            output_activation_fn=output_activation_fn,
            cumulate_att_w=cumulate_att_w,
            use_batch_norm=use_batch_norm,
            use_concate=use_concate,
            dropout_rate=dropout_rate,
            zoneout_rate=zoneout_rate,
            reduction_factor=reduction_factor,
        )

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

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
        ds_syb: Optional[torch.Tensor] = None,
        ds_syb_lengths: Optional[torch.Tensor] = None,

        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        # ALF
        """
        syllable = syllable[:, : syllable_lengths.max()]
        midi_syb = midi_syb[:, : midi_syb_lengths.max()]
        beat_syb = beat_syb[:, : beat_syb_lengths.max()]
        #print(beat_syb)

        syllable_emb = self.syllable_input_layer(syllable)  # B, len, dim
        midi_syb_emb = self.midi_syb_input_layer(midi_syb)
        beat_syb_emb = self.beat_syb_input_layer(beat_syb)

        # h_ds = syllable_emb + midi_syb_emb + ds_syb_emb  # B, len, dim
        # h_ds = linear
        
        syllable_emb = torch.nn.utils.rnn.pack_padded_sequence(
            syllable_emb,
            syllable_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        midi_syb_emb = torch.nn.utils.rnn.pack_padded_sequence(
            midi_syb_emb,
            midi_syb_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        print(midi_syb_emb.shape)
        ds_syb_emb = torch.nn.utils.rnn.pack_padded_sequence(
            ds_syb_emb, ds_syb_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        
        hs_syllable, (_, _) = self.syb_encoder(syllable_emb)
        hs_midi_syb, (_, _) = self.midi_syb_encoder(midi_syb_emb)
        hs_beat_syb, (_, _) = self.tempo_syb_encoder(beat_syb_emb)

        h_ds = hs_syllable + hs_midi_syb + hs_beat_syb

        ds_alf = self.phoneme_segmentation_predictor(h_ds)  # B, len, 4

        #label, midi, beat, ds_alf, len_alf 
        beat, beat_lengths = self.phoneme_length_regulator(
            syllable, syllable_num, syllable_lengths, beat_syb, ds_alf, label_xml, label_xml_lengths
        )
        
        # ds = ds_alf
        tempo = beat_xml # beat / beat_lab / beat_xml
        #label_lengths = len_alf
        #midi_lengths = len_alf
        label = label_xml
        midi = midi_xml
        label_lengths = label_xml_lengths
        midi_lengths = midi_xml_lengths
        #print(beat[0])
        #print(tempo[0])

        seg_loss = self.seg_loss(beat.to(dtype=torch.float), tempo.to(dtype=torch.float)) * 0.02
        #seg_loss = self.ds_loss(ds_alf, ds)  # !!
        #print("SEG", seg_loss)
        # end ALF

        """
        #label = label_xml
        midi = midi_xml
        #tempo = beat_xml
        #label_lengths = label_xml_lengths
        midi_lengths = midi_xml_lengths
        

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        #label = label[:, : label_lengths.max()]  # for data-parallel
        batch_size = feats.size(0)

        #label_emb = self.encoder_input_layer(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_encoder_input_layer(midi)
        #tempo_emb = self.tempo_encoder_input_layer(tempo)

        #label_emb = torch.nn.utils.rnn.pack_padded_sequence(
        #    label_emb, label_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        #)
        midi_emb = torch.nn.utils.rnn.pack_padded_sequence(
            midi_emb, midi_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        #tempo_emb = torch.nn.utils.rnn.pack_padded_sequence(
        #    tempo_emb, midi_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        #)

        #hs_label, (_, _) = self.encoder(label_emb)
        hs_midi, (_, _) = self.midi_encoder(midi_emb)
        #hs_tempo, (_, _) = self.tempo_encoder(tempo_emb)

        #hs_label, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_label, batch_first=True)
        hs_midi, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_midi, batch_first=True)
        #hs_tempo, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_tempo, batch_first=True)
        """
        if self.midi_embed_integration_type == "add":
            hs = hs_label + hs_midi + hs_tempo
            hs = F.leaky_relu(self.midi_projection(hs))
        else:
            hs = torch.cat((hs_label, hs_midi, hs_tempo), dim=-1)
            hs = F.leaky_relu(self.midi_projection(hs))
        # integrate spk & lang embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        """
        # forward duration predictor and length regulator
        #d_masks = make_pad_mask(label_lengths).to(hs.device)

        #after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)
        print("TEXT", text.shape)
        print("MIDI", midi.shape)

        after_outs, before_outs, logits, att_ws = self.dec(text, text_lengths, midi)
        print(before_outs.shape)
        print(att_ws.shape)
        #d_outs = self.duration_predictor(hs, d_masks)  # (B, T_text)
        #hs = self.length_regulator(hs, ds)  # (B, seq_len, eunits)

        hs_emb = torch.nn.utils.rnn.pack_padded_sequence(
            hs, feats_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )

        zs, (_, _) = self.decoder(hs_emb)
        zs, _ = torch.nn.utils.rnn.pad_packed_sequence(zs, batch_first=True)

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # postnet -> (B, T_feats//r * r, odim)
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

        # calculate loss values
        ilens = label_lengths
        l1_loss, duration_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens
        )
        #print(ds[0])
        #print(d_outs[0])
        #print("DUR", duration_loss)
        loss = l1_loss + duration_loss + seg_loss
        #loss = l1_loss + duration_loss
        stats = dict(
            loss=loss.item(), l1_loss=l1_loss.item(), duration_loss=duration_loss.item(), seg_loss=seg_loss.item()
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if flag_IsValid is False:
            # training stage
            return loss, stats, weight
        else:
            # validation stage
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
        ds_syb: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:


        # ALF

        #syllable = syllable[:, : syllable_lengths.max()]
        #midi_syb = midi_syb[:, : midi_syb_lengths.max()]
        #beat_syb = midi_syb[:, : beat_syb_lengths.max()]
        #print(beat_syb)

        syllable_emb = self.syllable_input_layer(syllable)  # B, len, dim
        midi_syb_emb = self.midi_syb_input_layer(midi_syb)
        beat_syb_emb = self.beat_syb_input_layer(beat_syb)

        # h_ds = syllable_emb + midi_syb_emb + ds_syb_emb  # B, len, dim
        # h_ds = linear
        """
        syllable_emb = torch.nn.utils.rnn.pack_padded_sequence(
            syllable_emb,
            syllable_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        midi_syb_emb = torch.nn.utils.rnn.pack_padded_sequence(
            midi_syb_emb,
            midi_syb_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        print(midi_syb_emb.shape)
        ds_syb_emb = torch.nn.utils.rnn.pack_padded_sequence(
            ds_syb_emb, ds_syb_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        """
        hs_syllable, (_, _) = self.syb_encoder(syllable_emb)
        hs_midi_syb, (_, _) = self.midi_syb_encoder(midi_syb_emb)
        hs_beat_syb, (_, _) = self.tempo_syb_encoder(beat_syb_emb)

        h_ds = hs_syllable + hs_midi_syb + hs_beat_syb

        ds_alf = self.phoneme_segmentation_predictor(h_ds)  # B, len, 4

        #label, midi, beat, ds_alf, len_alf 
        beat, beat_lengths = self.phoneme_length_regulator(
            syllable, syllable_num, syllable_lengths, beat_syb, ds_alf, label_xml, label_xml_lengths, is_inference=True
        )

        # ds = ds_alf
        tempo = torch.round(beat).to(torch.int64) # beat / beat_lab / beat_xml
        for i in range(len(tempo)):
            for j in range(len(tempo[i])):
                print(j, beat[i][j].item(), tempo[i][j].item(), beat_lab[i][j].item(), beat_xml[i][j].item())
        #label_lengths = len_alf
        #midi_lengths = len_alf
        label = label_xml
        midi = midi_xml
        #label_lengths = label_xml_lengths
        #midi_lengths = midi_xml_lengths
        #print(beat[0])
        #print(tempo[0])

        #seg_loss = self.seg_loss(beat.to(dtype=torch.float), tempo.to(dtype=torch.float)) * 0.02
        #seg_loss = self.ds_loss(ds_alf, ds)  # !!
        #print("SEG", seg_loss)
        # end ALF

        #label = label_xml
        #midi = midi_xml
        #tempo = beat_xml

        label_emb = self.encoder_input_layer(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_encoder_input_layer(midi)
        tempo_emb = self.tempo_encoder_input_layer(tempo)

        hs_label, (_, _) = self.encoder(label_emb)
        hs_midi, (_, _) = self.midi_encoder(midi_emb)
        hs_tempo, (_, _) = self.tempo_encoder(tempo_emb)

        if self.midi_embed_integration_type == "add":
            hs = hs_label + hs_midi + hs_tempo
            hs = F.leaky_relu(self.midi_projection(hs))
        else:
            hs = torch.cat((hs_label, hs_midi, hs_tempo), dim=-1)
            hs = F.leaky_relu(self.midi_projection(hs))
        # integrate spk & lang embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = None  # make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)

        hs = self.length_regulator(hs, d_outs_int)  # (B, T_feats, adim)
        zs, (_, _) = self.decoder(hs)

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, None, None  # outs, probs, att_ws

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
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

        return
