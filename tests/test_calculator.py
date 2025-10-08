import datetime
from pathlib import Path
import pandas as pd
import pytest
import unittest
from io import StringIO
from unittest.mock import Mock, patch, PropertyMock
from unittest.mock import patch, MagicMock
from io import StringIO
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
            
# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")

class TestCalculatorSaveHistoryException(unittest.TestCase):

    @patch('builtins.input', side_effect=['exit'])
    @patch('app.calculator.Calculator.save_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_save_history_exception(self, mock_stdout, mock_save_history, mock_input):
        # Make save_history raise an exception
        mock_save_history.side_effect = Exception("Disk full")

        calculator_repl()
    
        output = mock_stdout.getvalue()
        self.assertIn("Warning: Could not save history: Disk full", output)
        self.assertIn("Goodbye!", output)

    if __name__ == '__main__':
        unittest.main()

    @patch('builtins.input', side_effect=['history', 'exit'])
    @patch('app.calculator.Calculator.show_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_empty_history(self, mock_stdout, mock_show_history, mock_input):
        # Make show_history return an empty list
        mock_show_history.return_value = []

        # Run the REPL
        calculator_repl()

        output = mock_stdout.getvalue()

        # Assert it prints "No calculations in history"
        self.assertIn("No calculations in history", output)
        # REPL still exits
        self.assertIn("Goodbye!", output)

    @patch('builtins.input', side_effect=['history', 'exit'])
    @patch('app.calculator.Calculator.show_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_history_with_entries(self, mock_stdout, mock_show_history, mock_input):
        # Return a sample history
        mock_show_history.return_value = [
            "2 + 3 = 5",
            "10 / 2 = 5"
        ]

        calculator_repl()

        output = mock_stdout.getvalue()

        # Assert that the history is printed correctly
        self.assertIn("Calculation History:", output)
        self.assertIn("1. 2 + 3 = 5", output)
        self.assertIn("2. 10 / 2 = 5", output)

    @patch('builtins.input', side_effect=['undo', 'exit'])
    @patch('app.calculator.Calculator.undo')
    @patch('sys.stdout', new_callable=StringIO)
    def test_undo_success(self, mock_stdout, mock_undo, mock_input):
        # Simulate undo returning True
        mock_undo.return_value = True

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Operation undone", output)
        self.assertIn("Goodbye!", output)
    
    @patch('builtins.input', side_effect=['clear', 'exit'])
    @patch('app.calculator.Calculator.clear_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_clear_history(self, mock_stdout, mock_clear_history, mock_input):
        # Run the REPL
        calculator_repl()

        # Check that clear_history was called once
        mock_clear_history.assert_called_once()

        # Capture output and verify printed message
        output = mock_stdout.getvalue()
        self.assertIn("History cleared", output)
        self.assertIn("Goodbye!", output)

    @patch('builtins.input', side_effect=['undo', 'exit'])
    @patch('app.calculator.Calculator.undo')
    @patch('sys.stdout', new_callable=StringIO)
    def test_undo_nothing_to_undo(self, mock_stdout, mock_undo, mock_input):
        mock_undo.return_value = False  # simulate nothing to undo

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Nothing to undo", output)
        self.assertIn("Goodbye!", output)
    
    @patch('builtins.input', side_effect=['redo', 'exit'])
    @patch('app.calculator.Calculator.redo')
    @patch('sys.stdout', new_callable=StringIO)
    def test_redo_success(self, mock_stdout, mock_redo, mock_input):
        mock_redo.return_value = True  # simulate redo succeeds

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Operation redone", output)
        self.assertIn("Goodbye!", output)
    
    @patch('builtins.input', side_effect=['redo', 'exit'])
    @patch('app.calculator.Calculator.redo')
    @patch('sys.stdout', new_callable=StringIO)
    def test_redo_nothing_to_redo(self, mock_stdout, mock_redo, mock_input):
        mock_redo.return_value = False  # simulate nothing to redo

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Nothing to redo", output)
        self.assertIn("Goodbye!", output)
    
    @patch('builtins.input', side_effect=['save', 'exit'])
    @patch('app.calculator.Calculator.save_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_save_success(self, mock_stdout, mock_save_history, mock_input):
        # Simulate successful save (no exception)
        mock_save_history.return_value = None

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("History saved successfully", output)
        self.assertIn("Goodbye!", output)

    @patch('builtins.input', side_effect=['save', 'exit'])
    @patch('app.calculator.Calculator.save_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_save_failure(self, mock_stdout, mock_save_history, mock_input):
        # Make save_history raise an exception
        mock_save_history.side_effect = Exception("Disk full")

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Error saving history: Disk full", output)
        self.assertIn("Goodbye!", output)

    @patch('builtins.input', side_effect=['load', 'exit'])
    @patch('app.calculator.Calculator.load_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_load_success(self, mock_stdout, mock_load_history, mock_input):
        # Simulate successful load (no exception)
        mock_load_history.return_value = None

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("History loaded successfully", output)
        self.assertIn("Goodbye!", output)
    
    @patch('builtins.input', side_effect=['load', 'exit'])
    @patch('app.calculator.Calculator.load_history')
    @patch('sys.stdout', new_callable=StringIO)
    def test_load_failure(self, mock_stdout, mock_load_history, mock_input):
        # Make load_history raise an exception
        mock_load_history.side_effect = Exception("File not found")

        calculator_repl()

        output = mock_stdout.getvalue()
        self.assertIn("Error loading history: File not found", output)
        self.assertIn("Goodbye!", output)