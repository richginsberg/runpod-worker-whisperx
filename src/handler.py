import runpod
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import rp_whisperx

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
MODEL = rp_whisperx.WhisperX_worker()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    with rp_debugger.LineTimer('download_step'):
        job_input['audio'] = download_files_from_urls(job['id'], [job_input['audio']])[0]

    with rp_debugger.LineTimer('process_step'):
        transcription_results = MODEL.process(
            audio=job_input["audio"],
            compute_type=job_input["compute_type"],
            batch_size=job_input["batch_size"],
            language=job_input["language"]
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return transcription_results


runpod.serverless.start({"handler": handler})
