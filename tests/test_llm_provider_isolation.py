import unittest
from unittest.mock import patch

from layer2_decision import agent, corrector


class AgentProviderIsolationTests(unittest.TestCase):
    def setUp(self):
        self.perception = {"words": [], "silences": []}

    def test_explicit_provider_does_not_replace_default(self):
        with (
            patch.object(agent, "LLM_PROVIDER", "claude"),
            patch.object(agent, "_call_openai", return_value={"provider": "openai"}) as openai,
            patch.object(agent, "_call_claude", return_value={"provider": "claude"}) as claude,
        ):
            self.assertEqual(
                agent.call_llm(self.perception, provider="openai"),
                {"provider": "openai"},
            )
            self.assertEqual(
                agent.call_llm(self.perception),
                {"provider": "claude"},
            )

        openai.assert_called_once()
        claude.assert_called_once()

    def test_ollama_model_is_forwarded_per_call(self):
        with patch.object(agent, "_call_ollama", return_value={}) as ollama:
            agent.call_llm(
                self.perception,
                provider="ollama",
                ollama_model="llama3.2:3b",
            )

        ollama.assert_called_once()
        self.assertEqual(ollama.call_args.args[2], "llama3.2:3b")


class CorrectorProviderIsolationTests(unittest.TestCase):
    def test_explicit_provider_does_not_replace_default(self):
        with (
            patch.object(corrector, "LLM_PROVIDER", "claude"),
            patch.object(corrector, "_call_openai", return_value={"provider": "openai"}) as openai,
            patch.object(corrector, "_call_claude", return_value={"provider": "claude"}) as claude,
        ):
            self.assertEqual(
                corrector._call_llm("system", "user", provider="openai"),
                {"provider": "openai"},
            )
            self.assertEqual(
                corrector._call_llm("system", "user"),
                {"provider": "claude"},
            )

        openai.assert_called_once()
        claude.assert_called_once()


if __name__ == "__main__":
    unittest.main()
