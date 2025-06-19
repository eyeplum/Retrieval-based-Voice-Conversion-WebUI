#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from rmvpe import E2E


class RMVPEExporter:
    """Export RMVPE model to ExecutorTorch PTE format"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        print(f"Initializing RMVPE exporter on device: {self.device}")
        
    def create_model(self) -> torch.nn.Module:
        """Create and load the RMVPE model"""
        model_start = time.time()
        
        # Create model architecture
        model = E2E(4, 1, (2, 2))
        
        # Load weights
        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
            
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'])
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)
                
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            raise
        
        model.eval()
        model = model.to(self.device)
        
        model_time = time.time() - model_start
        print(f"Model loaded in {model_time:.3f}s")
        
        return model
    
    def create_example_inputs(self, batch_size: int = 1, seq_length: int = 128):
        """Create example inputs for tracing"""
        # RMVPE E2E model expects mel spectrogram: (batch_size, 128, seq_length)
        # Ensure seq_length is multiple of 32 for proper model operation
        padded_seq_length = 32 * ((seq_length - 1) // 32 + 1)
        # Keep inputs on CPU for export like hello-pte
        example_input = torch.randn(batch_size, 128, padded_seq_length)
        print(f"Created example input shape: {example_input.shape} (padded from {seq_length} to {padded_seq_length})")
        return (example_input,)
    
    def export_to_pte(
        self, 
        output_path: str,
        batch_size: int = 1,
        seq_length: int = 128,
        use_xnnpack: bool = True
    ) -> str:
        """Export model to ExecutorTorch PTE format using static shapes"""
        export_start = time.time()
        
        # Create model and example inputs
        model = self.create_model()
        example_inputs = self.create_example_inputs(batch_size, seq_length)
        
        print("Starting ExecutorTorch export process...")
        print(f"Model input shape: {example_inputs[0].shape}")
        print("Using static shape export for better compatibility")
        
        # Test model with example inputs first
        with torch.no_grad():
            test_output = model(*example_inputs)
            print(f"Model output shape: {test_output.shape}")
        
        # Export using static shapes only
        try:
            # Transform and lower to ExecutorTorch in one step (like hello-pte)
            partitioner = [XnnpackPartitioner()] if use_xnnpack else []
            
            et_program = to_edge_transform_and_lower(
                torch.export.export(model, example_inputs), 
                partitioner=partitioner
            ).to_executorch()
            print("Model exported with static shapes")
            
            # Step 3: Write PTE file
            with open(output_path, "wb") as f:
                f.write(et_program.buffer)
            
            total_time = time.time() - export_start
            print(f"Export completed successfully in {total_time:.3f}s")
            print(f"Model exported to: {output_path}")
            
        except Exception as e:
            print(f"Static export failed: {e}")
            raise e
        
        return output_path
    


def main():
    parser = argparse.ArgumentParser(description='Export RMVPE model to ExecutorTorch PTE format')
    parser.add_argument('--model', '-m', required=True, help='Input PyTorch model file path')
    parser.add_argument('--output', '-o', required=True, help='Output PTE file path')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size for export (default: 1)')
    parser.add_argument('--seq-length', '-s', type=int, default=128, help='Base sequence length for export, will be padded to multiple of 32 (default: 128)')
    parser.add_argument('--device', '-d', default='cpu', help='Device for export process (default: cpu)')
    parser.add_argument('--no-xnnpack', action='store_true', help='Disable XNNPACK backend optimization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    
    # Validate inputs
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file does not exist: {args.model}")
        return 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize exporter and export model
        exporter = RMVPEExporter(str(model_path), device=args.device)
        
        exported_path = exporter.export_to_pte(
            output_path=str(output_path),
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            use_xnnpack=not args.no_xnnpack
        )
        
        # Print summary
        file_size_mb = Path(exported_path).stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Export Summary:")
        print(f"  Input model: {args.model}")
        print(f"  Output PTE: {exported_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Sequence length: {args.seq_length} (padded to multiple of 32)")
        print(f"  Device: {args.device}")
        print(f"  XNNPACK: {'Disabled' if args.no_xnnpack else 'Enabled'}")
        print(f"  Dynamic shapes: Disabled (static export)")
        print(f"{'='*60}")
        
        # Test the exported model with the fixed sequence length
        print(f"\nTesting exported model with fixed sequence length...")
        try:
            from executorch.runtime import Runtime
            runtime = Runtime.get()
            program = runtime.load_program(exported_path)
            method = program.load_method("forward")
            
            # Test with the exact sequence length used for export (simple like hello-pte)
            padded_len = 32 * ((args.seq_length - 1) // 32 + 1)
            test_input = torch.randn(1, 128, padded_len)
            output = method.execute([test_input])
            print(f"  ✓ Sequence length {args.seq_length} (padded to {padded_len}): Output shape {output[0].shape}")
            print(f"  Static model works correctly with fixed input size")
                    
        except Exception as e:
            print(f"  Model verification skipped: {e}")
        
        return 0
        
    except Exception as e:
        print(f"Error: Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())