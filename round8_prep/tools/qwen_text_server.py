"""Use SGLang's existing programmatic text-only switch; no model replacement."""
import sys


def configure_text(args):
    assert not args.language_only and not args.encoder_only
    args.enable_multimodal = False
    return args


if __name__ == '__main__':
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    args = configure_text(prepare_server_args(sys.argv[1:]))
    print('ROUND8_TEXT_ENTRY enable_multimodal=False; model architecture unchanged', flush=True)
    run_server(args)
