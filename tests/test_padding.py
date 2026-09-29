"""Regression checks for padding-independent training and generation.

Run with python -m unittest discover -s tests -v. Set LOOPING_TEST_DEVICE=cuda
to exercise the GPU attention implementation with the same checks.
"""

import contextlib
import io
import os
import unittest
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

from core_train import initialize_model
from data.loop_stargraph import LoopStarGraphDataModule, Tokenizer
from model_gpt import GPT, GPTConfig


class PaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = torch.device(os.environ.get("LOOPING_TEST_DEVICE", "cpu"))
        cls.fabric = L.Fabric(accelerator=cls.device.type, devices=1, precision="32-true")
        cls.tokenizer = Tokenizer(30)

    def setUp(self):
        torch.manual_seed(1234)
        self.examples = [
            self.encode("0,1|1,2/0,2=0,1,2"),
            self.encode("0,1|1,2/0,2=0,1,0,1,0,1,2"),
            self.encode("0,1|1,2/0,2=0,1,0,1,2"),
        ]

    def encode(self, text):
        return torch.tensor(self.tokenizer.tokenize(text)[0], dtype=torch.long)

    def make_model(self, manual=False, response_only=True):
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            n_layer=2, n_head=2, n_embd=64, block_size=256,
            dropout=0.0, loss_on_response_only=response_only,
        )
        model = GPT(config)
        if manual:
            for block in model.model.transformer.blocks:
                block.attn.flash = False
                block.attn.register_buffer(
                    "bias", torch.ones(256, 256).tril().view(1, 1, 256, 256)
                )
        with contextlib.redirect_stdout(io.StringIO()):
            model.setup_fabric(self.fabric)
        model.eval()
        return model

    def test_initializer_forwards_padding_and_response_loss(self):
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/defaults.yaml")
        config.model.n_layer = 1
        config.model.n_head = 2
        config.model.n_embd = 64
        config.model.block_size = 256
        config.model.vocab_size = self.tokenizer.vocab_size
        config.model.pad_token_id = self.tokenizer.pad_token_id
        config.model.loss_on_response_only = False
        config.trainer.compile = False
        with contextlib.redirect_stdout(io.StringIO()):
            model = initialize_model(self.fabric, config, self.tokenizer, initialize_optimizer=False)
        self.assertEqual(model.config.pad_token_id, self.tokenizer.pad_token_id)
        self.assertEqual(model.model.config.pad_token_id, self.tokenizer.pad_token_id)
        self.assertFalse(model.config.loss_on_response_only)
        # Preserve the released model's effective 8/3 SwiGLU expansion.
        self.assertIsNone(model.config.expansion_factor)

    def test_logits_ignore_padding_and_batch_neighbors(self):
        for manual in (False, True):
            with self.subTest(manual_attention=manual):
                model = self.make_model(manual=manual)
                unpadded = self.examples[0][:-1].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    expected = model.model(unpadded)
                    for count in (1, 60, 90, 130):
                        padded = torch.nn.functional.pad(
                            unpadded, (count, 0), value=self.tokenizer.pad_token_id
                        )
                        logits = model.model(padded)
                        self.assertTrue(torch.isfinite(logits).all())
                        torch.testing.assert_close(
                            logits[:, count:], expected, atol=2e-5, rtol=2e-5
                        )
                    batch, _ = LoopStarGraphDataModule.collate_fn(self.examples, self.tokenizer)
                    logits = model.model(batch[:, :-1].to(self.device))
                    self.assertTrue(torch.isfinite(logits).all())
                    for i, example in enumerate(self.examples):
                        solo = model.model(example[:-1].unsqueeze(0).to(self.device))
                        count = batch.size(1) - example.numel()
                        torch.testing.assert_close(
                            logits[i:i+1, count:], solo, atol=2e-5, rtol=2e-5
                        )

    def test_loss_and_gradients_ignore_padding(self):
        for manual in (False, True):
            with self.subTest(manual_attention=manual):
                model = self.make_model(manual=manual)
                example = self.examples[0].unsqueeze(0).to(self.device)
                prompt_len = int((example[0] == self.tokenizer.encoder["="]).nonzero()[0]) + 1
                losses, gradients = [], []
                for count in (0, 90):
                    model.model.zero_grad(set_to_none=True)
                    padded = torch.nn.functional.pad(
                        example, (count, 0), value=self.tokenizer.pad_token_id
                    )
                    loss = model.compute_loss(
                        padded, backpropagate=True,
                        prompt_len=torch.tensor([prompt_len + count], device=self.device),
                    )
                    losses.append(loss)
                    gradients.append({
                        name: p.grad.clone() for name, p in model.model.named_parameters()
                        if p.grad is not None
                    })
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients[-1].values()))
                    pad_grad = model.model.token_embedding.weight.grad[self.tokenizer.pad_token_id]
                    self.assertEqual(float(pad_grad.abs().max()), 0.0)
                torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=2e-5)
                for name in gradients[0]:
                    torch.testing.assert_close(
                        gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-5
                    )

    def test_response_targets_include_first_walk_token_and_eos(self):
        model = self.make_model()
        batch, prompt_len = LoopStarGraphDataModule.collate_fn(self.examples, self.tokenizer)
        captured = []

        def capture_targets(module, args, kwargs):
            captured.append(kwargs["targets"].detach().cpu())

        hook = model.model.module.register_forward_pre_hook(capture_targets, with_kwargs=True)
        try:
            model.compute_loss(
                batch.to(self.device), backpropagate=False, prompt_len=prompt_len.to(self.device)
            )
        finally:
            hook.remove()
        for i, example in enumerate(self.examples):
            equals = int((example == self.tokenizer.encoder["="]).nonzero()[0])
            targets = captured[0][i]
            torch.testing.assert_close(targets[targets != -100], example[equals + 1:])

    def test_full_sequence_loss_does_not_predict_first_token_from_padding(self):
        model = self.make_model(response_only=False)
        sample = self.examples[0].unsqueeze(0).to(self.device)
        padded = torch.nn.functional.pad(sample, (20, 0), value=self.tokenizer.pad_token_id)
        plain_loss = model.compute_loss(sample, backpropagate=False)
        padded_loss = model.compute_loss(padded, backpropagate=False)
        torch.testing.assert_close(plain_loss, padded_loss, atol=2e-5, rtol=2e-5)

    def test_greedy_generation_ignores_leading_padding(self):
        model = self.make_model()
        example = self.examples[0]
        prompt_len = int((example == self.tokenizer.encoder["="]).nonzero()[0]) + 1
        prompt = example[:prompt_len].unsqueeze(0).to(self.device)
        expected = model.generate(prompt, max_new_tokens=8, do_sample=False)[:, prompt_len:]
        for count in (60, 90, 130):
            padded = torch.nn.functional.pad(prompt, (count, 0), value=self.tokenizer.pad_token_id)
            actual = model.generate(padded, max_new_tokens=8, do_sample=False)[:, count + prompt_len:]
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
