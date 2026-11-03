"""Fail for dependency conflicts except the documented GPU NCCL metadata pin."""
import subprocess
import sys


def main():
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    allowed = 'torch 2.6.0+cu126 has requirement nvidia-nccl-cu12==2.21.5; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-nccl-cu12 2.25.1.'
    if result.returncode and lines != [allowed]:
        raise RuntimeError(f"Unexpected dependency conflict: {result.stdout} {result.stderr}")
    print(result.stdout or "Dependencies checked")
    if lines == [allowed]:
        print("Known NCCL metadata exception: shared 2.25.1 runtime; single-GPU only. See MODEL_NOTES.md.")


if __name__ == "__main__":
    main()
