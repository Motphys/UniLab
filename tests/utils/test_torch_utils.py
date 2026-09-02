"""Tests for unilab.utils.tensor."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from unilab.utils.tensor import to_numpy, to_torch


class TestToTorch:
    """Tests for to_torch."""

    def test_tensor_input_cpu(self) -> None:
        x = torch.randn(4, 8)
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_input_cuda(self) -> None:
        x = torch.randn(4, 8)
        result = to_torch(x, "cuda")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_tensor_input_mps(self) -> None:
        x = torch.randn(4, 8)
        result = to_torch(x, "mps")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "mps"

    def test_numpy_array_input_cpu(self) -> None:
        x = np.random.randn(4, 8).astype(np.float32)
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        assert result.shape == (4, 8)
        np.testing.assert_array_almost_equal(result.numpy(), x)

    def test_numpy_array_input_2d(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert result.dtype == torch.float32
        np.testing.assert_array_almost_equal(result.numpy(), x)

    def test_numpy_array_input_1d(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result.numpy(), x)

    def test_dlpack_tensor_input(self) -> None:
        x = torch.randn(3, 5)
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        np.testing.assert_array_almost_equal(result.numpy(), x.numpy())

    @pytest.mark.parametrize(
        "error_type",
        [
            AttributeError,
            BufferError,
            NotImplementedError,
            TypeError,
            ValueError,
            RuntimeError,
        ],
    )
    def test_dlpack_expected_failure_logs_and_falls_back(
        self,
        error_type: type[Exception],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class ArrayWithDLPackFallback:
            def __dlpack__(self):
                raise AssertionError("torch.from_dlpack should own protocol invocation")

            def __array__(self, dtype=None, copy=None):
                del copy
                return np.asarray([1.0, 2.0], dtype=dtype)

        failure = error_type("broken dlpack")

        def fail_dlpack(_value):
            raise failure

        monkeypatch.setattr(torch, "from_dlpack", fail_dlpack)
        with caplog.at_level(logging.WARNING, logger="unilab.utils.tensor"):
            result = to_torch(ArrayWithDLPackFallback(), "cpu")

        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
        assert result.dtype == torch.float32
        assert "dlpack conversion failed" in caplog.text
        assert "broken dlpack" in caplog.text

    def test_partial_dlpack_protocol_uses_numpy_fallback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class PartialDLPack:
            def __dlpack__(self):
                raise AssertionError("missing __dlpack_device__ must fail first")

            def __array__(self, dtype=None, copy=None):
                del copy
                return np.asarray([1.0, 2.0], dtype=dtype)

        with caplog.at_level(logging.WARNING, logger="unilab.utils.tensor"):
            result = to_torch(PartialDLPack(), "cpu")

        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
        assert "has no attribute '__dlpack_device__'" in caplog.text

    def test_dlpack_unexpected_failure_propagates_without_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class DLPackOnly:
            def __dlpack__(self):
                raise AssertionError("torch.from_dlpack should own protocol invocation")

            def __array__(self, dtype=None, copy=None):
                del dtype, copy
                raise AssertionError("unexpected errors must not enter the numpy fallback")

        failure = OSError("unexpected dlpack failure")

        def fail_dlpack(_value):
            raise failure

        monkeypatch.setattr(torch, "from_dlpack", fail_dlpack)
        with pytest.raises(OSError) as caught:
            to_torch(DLPackOnly(), "cpu")

        assert caught.value is failure
        assert not caplog.records

    def test_list_input(self) -> None:
        x = [1.0, 2.0, 3.0]
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_nested_list_input(self) -> None:
        x = [[1.0, 2.0], [3.0, 4.0]]
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_tuple_input(self) -> None:
        x = (1.0, 2.0, 3.0)
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_scalar_input(self) -> None:
        x = 3.14
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == ()
        assert result.item() == pytest.approx(3.14)

    def test_int_list_input(self) -> None:
        x = [1, 2, 3]
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_device_as_torch_device(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        device = torch.device("cpu")
        result = to_torch(x, device)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_empty_array(self) -> None:
        x = np.array([])
        result = to_torch(x, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (0,)


class TestToNumpy:
    """Tests for to_numpy."""

    def test_numpy_array_input(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result is x

    def test_torch_tensor_input_cpu(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_torch_tensor_input_requires_grad(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_tensor_input_cuda(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_list_input(self) -> None:
        x = [1.0, 2.0, 3.0]
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_tuple_input(self) -> None:
        x = (1.0, 2.0, 3.0)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_scalar_input(self) -> None:
        x = 3.14
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert result.item() == pytest.approx(3.14)

    def test_2d_tensor_input(self) -> None:
        x = torch.randn(4, 8)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 8)
        np.testing.assert_array_almost_equal(result, x.numpy())

    def test_empty_tensor(self) -> None:
        x = torch.tensor([])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_nested_list_input(self) -> None:
        x = [[1.0, 2.0], [3.0, 4.0]]
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)
