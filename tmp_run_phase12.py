from tests.test_phase12 import test_e2e_generate_response_shape

try:
    test_e2e_generate_response_shape()
    print('function completed successfully')
except Exception as exc:
    import traceback
    traceback.print_exc()
    print('function raised exception:', type(exc).__name__, exc)
