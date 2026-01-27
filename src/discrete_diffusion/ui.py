"""Gradio UI for discrete diffusion text generation."""

import argparse

import gradio as gr
import torch

from .pipeline import DiffusionPipeline


class DiffusionApp:
    """Wrapper class for Gradio app with model state."""

    def __init__(self, model_path: str, device: str = None):
        """Initialize the app with a loaded model."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model from {model_path}...")
        self.pipe = DiffusionPipeline.from_pretrained(model_path, device=self.device)
        print(f"Model loaded with {self.pipe.model.get_num_params()/1e6:.2f}M parameters")
        print(f"Using device: {self.device}")

    def generate_text(
        self,
        initial_sequence: str,
        num_steps: int,
    ):
        """
        Generate text with streaming visualization.

        This is a generator function that yields intermediate results
        at each denoising step. Gradio will automatically update all
        output components with each yield.

        Args:
            initial_sequence: Input text with '_' indicating positions to infill.
                            Non-underscore characters will be used as context.
            num_steps: Number of denoising steps

        Yields:
            Tuple of (text, progress, step_info, noise_level)
        """
        try:
            # Validate initial sequence length
            if len(initial_sequence) != 256:
                error_msg = f"Initial sequence must be exactly 256 characters long. Current length: {len(initial_sequence)}"
                yield (error_msg, 0, "Error", "N/A")
                return

            # Stream generation with visualization
            for step, text, noise_level in self.pipe.generate_streaming(
                num_steps=num_steps,
                initial_sequence=initial_sequence,
            ):
                # Calculate progress (0 to 1)
                progress = step / num_steps

                # Format step info
                step_info = f"Step {step}/{num_steps}"

                # Format noise level
                noise_info = f"{noise_level:.6f}"

                # Yield all outputs: text, progress, step_info, noise_level
                yield (text, progress, step_info, noise_info)

        except Exception as e:
            yield (f"Error generating text: {str(e)}", 0, "Error", "N/A")


def create_interface(app: DiffusionApp):
    """Create the Gradio interface."""

    with gr.Blocks(title="Discrete Diffusion Text Infilling") as demo:
        gr.Markdown(
            """
            # Discrete Diffusion Text Infilling

            Infill text samples using a discrete diffusion model. Adjust the parameters below to control generation.

            - **Initial Sequence**: The text to infill. Must contain exactly 256 characters, where '_' indicates a missing character to be infilled.
            - **Denoising Steps**: More steps = higher quality but slower
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                gr.Markdown("### Input")

                initial_sequence = gr.Textbox(
                    label="Initial Sequence",
                    lines=4,
                    max_lines=6,
                    placeholder="Enter text with '_' for missing characters...",
                    info="Use '_' to mark positions to infill.",
                    value="Once upon a time, there lived " \
                    "________________________________________________________________" \
                    "________________________________________________________________" \
                    "________________________________________________________________" \
                    "_______ until the dragon ate them.",
                    elem_id="initial_sequence_input",
                )

                char_count = gr.Textbox(
                    label="Character Count (must be exactly 256)",
                    value="256 / 256 ✓",
                    interactive=False,
                    max_lines=1,
                )

                num_steps = gr.Slider(
                    minimum=32,
                    maximum=512,
                    value=256,
                    step=32,
                    label="Denoising Steps",
                    info="More steps = better quality but slower"
                )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                # Output display
                gr.Markdown("### Output")

                # Progress indicators
                with gr.Row():
                    step_display = gr.Textbox(
                        label="Current Step",
                        value="Ready",
                        interactive=False,
                        scale=1,
                    )
                    noise_display = gr.Textbox(
                        label="Noise Level (σ)",
                        value="N/A",
                        interactive=False,
                        scale=1,
                    )

                progress_bar = gr.Slider(
                    label="Denoising Progress",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                    show_label=True,
                )

                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=5,
                    max_lines=10,
                    placeholder="Generated text will appear here...",
                    interactive=False,
                )

        # Connect button to generation function
        generate_btn.click(
            fn=app.generate_text,
            inputs=[initial_sequence, num_steps],
            outputs=[output_text, progress_bar, step_display, noise_display],
        )

        # Auto-adjust text to 256 characters by adding/removing underscores
        def auto_adjust_length(text):
            """
            Automatically adjust text to exactly 256 characters by intelligently
            adding or removing underscores from the longest contiguous sequence.
            """
            import re

            length = len(text)

            # If already 256, no adjustment needed
            if length == 256:
                char_count_msg = f"{length} / 256 ✓"
                return text, char_count_msg

            # Find all contiguous sequences of underscores
            underscore_sequences = list(re.finditer(r'_+', text))

            if length < 256:
                # Need to add underscores
                needed = 256 - length

                if underscore_sequences:
                    # Find the longest sequence
                    longest = max(underscore_sequences, key=lambda m: len(m.group()))
                    # Insert underscores at the end of the longest sequence
                    start, end = longest.span()
                    adjusted_text = text[:end] + ('_' * needed) + text[end:]
                else:
                    # No underscores found, add them at the end
                    adjusted_text = text + ('_' * needed)

                char_count_msg = f"256 / 256 ✓ (added {needed})"

            else:  # length > 256
                # Need to remove underscores
                to_remove = length - 256

                if underscore_sequences:
                    # Find the longest sequence
                    longest = max(underscore_sequences, key=lambda m: len(m.group()))
                    start, end = longest.span()
                    sequence_length = end - start

                    if sequence_length >= to_remove:
                        # Remove from the end of the longest sequence
                        new_end = end - to_remove
                        adjusted_text = text[:new_end] + text[end:]
                    else:
                        # Longest sequence is too short, remove it entirely and continue
                        adjusted_text = text[:start] + text[end:]
                        # Recursively adjust if still too long
                        if len(adjusted_text) > 256:
                            adjusted_text, _ = auto_adjust_length(adjusted_text)
                else:
                    # No underscores to remove, truncate from the end
                    adjusted_text = text[:256]

                char_count_msg = f"256 / 256 ✓ (removed {to_remove})"

            return adjusted_text, char_count_msg

        # Update character count on change (but don't modify text while typing)
        def update_count_only(text):
            length = len(text)
            if length == 256:
                return f"{length} / 256 ✓"
            elif length < 256:
                return f"{length} / 256 (need {256 - length} more)"
            else:
                return f"{length} / 256 (remove {length - 256})"

        initial_sequence.change(
            fn=update_count_only,
            inputs=initial_sequence,
            outputs=char_count,
        )

        # Auto-adjust on blur (when user leaves the textbox) to avoid cursor jumping
        initial_sequence.blur(
            fn=auto_adjust_length,
            inputs=initial_sequence,
            outputs=[initial_sequence, char_count],
        )

    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch Gradio UI for text generation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (HF directory or .pth file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link (for remote access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)",
    )

    args = parser.parse_args()

    # Create app
    app = DiffusionApp(model_path=args.model, device=args.device)

    # Create interface
    demo = create_interface(app)

    # Launch
    print(f"\nLaunching Gradio interface on http://{args.server_name}:{args.port}")
    if args.share:
        print("Creating public share link...")

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
