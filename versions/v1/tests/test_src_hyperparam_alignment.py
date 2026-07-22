import ast
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'versions' / 'v1' / 'scripts'))
from profile_config import SRC_ORIGINAL, profile_values
from run_profile import build_command


EXPECTED = {
    'num_train_epochs': 40, 'learning_rate': 5e-5, 'batch_size': 4,
    'gradient_accumulation_steps': 4, 'weight_decay': 0.0, 'warmup_steps': 2000,
    'mixed_precision': 'no', 'score_dropout': 0.5, 'hidden_dropout': 0.0,
    'adapter_dropout': 0.3, 'lora_r': 8, 'lora_alpha': 32, 'lora_frequency': 1,
    'hd_frequency': 1, 'kl_loss_weight': 1.0, 'alternating_learning': 2,
    'fusion_alpha': 0.5, 'fusion_temperature': 0.5, 'fusion_type': 'text',
    'max_item_num': 30, 'max_token_num': 1024, 'skip_valid': 15, 'patient': 10,
    'seed': 42, 'num_workers': 1,
}


def test_src_original_profile_matches_src_parameters_defaults():
    assert SRC_ORIGINAL == EXPECTED
    source = (ROOT / 'src' / 'parameters.py').read_text()
    tree = ast.parse(source)
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'add_argument' or not node.args:
            continue
        option = ast.literal_eval(node.args[0])
        for keyword in node.keywords:
            if keyword.arg == 'default':
                try:
                    defaults[option.lstrip('-')] = ast.literal_eval(keyword.value)
                except ValueError:
                    pass
    for key, value in EXPECTED.items():
        assert defaults[key] == value


def test_legacy_profile_is_explicit_and_unknown_profile_fails():
    assert profile_values('legacy_tuned', 'Video_Games')['num_train_epochs'] == 12
    with pytest.raises(ValueError, match='Unknown v1 profile'):
        profile_values('unknown', 'Video_Games')


def test_profile_command_explicitly_records_all_key_parameters(tmp_path):
    class Args:
        dataset = 'Video_Games'; profile = 'src_original'; model = 'model'
        suffix = None; gpu = '7'; data_root = str(tmp_path); output_dir = str(tmp_path)
        gradient_checkpointing = False
    command = build_command(Args())
    assert '--profile' in command and 'src_original' in command
    for key, value in SRC_ORIGINAL.items():
        index = command.index(f'--{key}')
        assert command[index + 1] == str(value)
