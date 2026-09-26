import unittest

from pathlib import Path

from artifacts.round11.build_t2_attribution_native import build_command, validate_cache_flags


class BuildT2AttributionNativeTests(unittest.TestCase):
    def test_frozen_arm_flags_are_accepted(self):
        cache = "\n".join(
            [
                "CMAKE_BUILD_TYPE:STRING=Release",
                "CPUINFER_ENABLE_LTO:BOOL=OFF",
                "KTRANSFORMERS_USE_ASCEND_NPU:BOOL=ON",
                "LLAMA_ARM_DOTPROD:BOOL=ON",
                "LLAMA_ARM_FP16:BOOL=ON",
                "LLAMA_ARM_SVE:BOOL=OFF",
                "LLAMA_ARM_BF16:BOOL=OFF",
                "LLAMA_ARM_I8MM:BOOL=OFF",
                "LLAMA_NATIVE:BOOL=OFF",
                "PYTHON_EXECUTABLE:FILEPATH=/usr/local/python3.11.15/bin/python3",
            ]
        )
        result = validate_cache_flags(cache)
        self.assertEqual(result["LLAMA_ARM_DOTPROD"], "ON")
        self.assertEqual(result["LLAMA_ARM_SVE"], "OFF")

    def test_isa_drift_fails_closed(self):
        cache = "LLAMA_ARM_SVE:BOOL=ON\n"
        with self.assertRaisesRegex(ValueError, "cache flag"):
            validate_cache_flags(cache)

    def test_reused_cache_is_mounted_at_its_original_build_path(self):
        command = build_command(Path("/source"), Path("/candidate"), "v2")
        self.assertIn("/source:/round9/candidate-native-src:ro", command)
        self.assertIn("/candidate:/build", command)
        self.assertEqual(command[command.index("-w") + 1], "/build")
        self.assertIn("kt-round11-t2-attribution-build-v2", command)


if __name__ == "__main__":
    unittest.main()
