import subprocess
from unittest.mock import MagicMock
from unittest.mock import patch

from mhpy.utils.subprocess import run_cmd


class TestRunCmd:
    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_success(self, mock_run):
        command = "echo 'Hello World'"
        error_msg = "Echo failed"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        mock_run.assert_called_once_with(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @patch("mhpy.utils.subprocess.logger")
    @patch("mhpy.utils.subprocess.subprocess.run")
    @patch("mhpy.utils.subprocess.sys.exit")
    def test_run_cmd_failure(self, mock_exit, mock_run, mock_logger):
        """Test run_cmd with a failing command."""
        command = "false"
        error_msg = "Command failed"
        stderr_output = "Error: command not found"

        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=command, stderr=stderr_output)

        run_cmd(command, error_msg)

        mock_logger.error.assert_called_once()
        error_call = str(mock_logger.error.call_args)
        assert error_msg in error_call
        assert stderr_output in error_call

        mock_exit.assert_called_once_with(1)

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_with_complex_command(self, mock_run):
        command = "ls -la | grep test | wc -l"
        error_msg = "Pipeline failed"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == command

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_shell_true(self, mock_run):
        """Test that run_cmd uses shell=True."""
        command = "echo test"
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["shell"] is True

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_check_true(self, mock_run):
        command = "echo test"
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["check"] is True

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_captures_output(self, mock_run):
        command = "echo test"
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["stdout"] == subprocess.PIPE
        assert mock_run.call_args[1]["stderr"] == subprocess.PIPE

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_text_mode(self, mock_run):
        """Test that run_cmd uses text mode."""
        command = "echo test"
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["text"] is True

    @patch("mhpy.utils.subprocess.logger")
    @patch("mhpy.utils.subprocess.subprocess.run")
    @patch("mhpy.utils.subprocess.sys.exit")
    def test_run_cmd_empty_stderr(self, mock_exit, mock_run, mock_logger):
        command = "false"
        error_msg = "Command failed"

        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=command, stderr="")

        run_cmd(command, error_msg)

        mock_logger.error.assert_called_once()
        mock_exit.assert_called_once_with(1)

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_logs_command(self, mock_run):
        command = "custom_command --flag value"
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)
