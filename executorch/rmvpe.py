import argparse
from typing import List
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import pad_center
from scipy.signal import get_window
from librosa.filters import mel


logger = logging.getLogger(__name__)


class STFT(torch.nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, return_phase=False):
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )
        forward_transform = input_data.unfold(
            1, self.filter_length, self.hop_length
        ).permute(0, 2, 1)
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
        device=None,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half
        self.device = device

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )

        # Always use torch.stft for executorch compatibility
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(self, model_path: str, is_half: bool = False, device: str = None):
        init_start = time.time()
        self.resample_kernel = {}
        self.is_half = is_half
        
        # Configurable device handling
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize mel extractor with device
        mel_start = time.time()
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000, device=self.device
        ).to(self.device)
        mel_time = time.time() - mel_start
        logger.info(f"Mel extractor initialized in {mel_time:.3f}s")
        
        # Load model with backward compatibility
        model_start = time.time()
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        model_time = time.time() - model_start
        logger.info(f"Model loaded in {model_time:.3f}s")
        
        # Initialize cents mapping
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))
        
        init_time = time.time() - init_start
        logger.info(f"RMVPE initialization completed in {init_time:.3f}s")

    def _load_model(self, model_path: str):
        """Load model with backward compatibility for previous model files"""
        model = E2E(4, 1, (2, 2))
        
        # Load checkpoint with error handling for different formats
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            
            # Handle different checkpoint formats
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'])
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                # Assume it's a state dict
                model.load_state_dict(ckpt)
                
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
        
        model.eval()
        
        if self.is_half:
            model = model.half()
        else:
            model = model.float()
            
        return model

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            
            mel = mel.half() if self.is_half else mel.float()
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        infer_start = time.time()
        
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        
        # Mel extraction timing
        mel_start = time.time()
        mel = self.mel_extractor(
            audio.float().to(self.device).unsqueeze(0), center=True
        )
        mel_time = time.time() - mel_start
        
        # Model inference timing
        model_start = time.time()
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        model_time = time.time() - model_start
        
        if self.is_half:
            hidden = hidden.astype("float32")

        # Decoding timing
        decode_start = time.time()
        f0 = self.decode(hidden, thred=thred)
        decode_time = time.time() - decode_start
        
        total_time = time.time() - infer_start
        logger.info(f"Inference timing - Mel: {mel_time:.3f}s, Model: {model_time:.3f}s, Decode: {decode_time:.3f}s, Total: {total_time:.3f}s")
        
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided


def main():
    parser = argparse.ArgumentParser(description='RMVPE Pitch Estimation')
    parser.add_argument('--input', '-i', required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', help='Output f0 file path (optional)')
    parser.add_argument('--model', '-m', required=True, help='Model file path')
    parser.add_argument('--device', '-d', default=None, help='Device (cuda, cpu, mps, etc.)')
    parser.add_argument('--half', action='store_true', help='Use half precision')
    parser.add_argument('--threshold', '-t', type=float, default=0.03, help='Threshold for f0 detection')
    parser.add_argument('--plot', action='store_true', help='Plot f0 contour')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load audio
    try:
        import librosa
        import soundfile as sf
        
        audio_start = time.time()
        audio, sampling_rate = sf.read(args.input)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        
        audio_time = time.time() - audio_start
        logger.info(f"Audio loaded and resampled in {audio_time:.3f}s")
            
    except ImportError:
        logger.error("Please install librosa and soundfile: pip install librosa soundfile")
        return
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return
    
    # Initialize RMVPE
    rmvpe = RMVPE(args.model, is_half=args.half, device=args.device)
    
    # Infer f0
    f0 = rmvpe.infer_from_audio(audio, thred=args.threshold)
    
    # Calculate statistics
    f0_nonzero = f0[f0 > 0]
    if len(f0_nonzero) > 0:
        f0_mean = np.mean(f0_nonzero)
        f0_min = np.min(f0_nonzero)
        f0_max = np.max(f0_nonzero)
        voiced_ratio = len(f0_nonzero) / len(f0)
    else:
        f0_mean = f0_min = f0_max = 0
        voiced_ratio = 0
    
    # Print results
    print(f"F0 shape: {f0.shape}")
    print(f"Voiced frames: {len(f0_nonzero)}/{len(f0)} ({voiced_ratio:.1%})")
    if len(f0_nonzero) > 0:
        print(f"F0 mean: {f0_mean:.2f} Hz")
        print(f"F0 range: {f0_min:.2f} - {f0_max:.2f} Hz")
    
    # Save results
    if args.output:
        np.save(args.output, f0)
        print(f"F0 saved to {args.output}")
    
    # Plot f0 if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            time_frames = np.arange(len(f0)) * 160 / 16000  # 160 hop length, 16kHz sample rate
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            
            # First plot - full f0 contour
            ax1.plot(time_frames, f0, 'b-', linewidth=1, alpha=0.8)
            ax1.set_ylabel('F0 (Hz)')
            ax1.set_title('F0 Contour')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(f0_max * 1.1, 500))
            
            # Second plot - voiced regions only
            voiced_mask = f0 > 0
            if np.any(voiced_mask):
                ax2.plot(time_frames[voiced_mask], f0[voiced_mask], 'r.', markersize=2)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('F0 (Hz)')
                ax2.set_title('F0 Contour (Voiced Only)')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(f0_min * 0.9, f0_max * 1.1)
            
            plt.tight_layout()
            
            # Save plot if output specified
            if args.output:
                plot_path = args.output.replace('.npy', '_plot.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"F0 plot saved to {plot_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.error("Please install matplotlib for plotting: pip install matplotlib")
        except Exception as e:
            logger.error(f"Error creating plot: {e}")


if __name__ == "__main__":
    main()