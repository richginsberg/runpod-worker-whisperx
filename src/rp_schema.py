INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': True
    },
    'compute_type': {
        'type': str,
        'required': False,
        'default': 'float32'
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 16
    },
    'language': {
        'type': str,
        'required': False,
        'default': 'unknown'
    }
}