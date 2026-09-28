"""Check the inferential comparison and the actual configurable API payload."""
import importlib.util
from pathlib import Path
import httpx
import pytest
from jev_benchmark.jev import classify

spec=importlib.util.spec_from_file_location('publish_analysis',Path(__file__).parents[1]/'scripts/publish_analysis.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_paired_bootstrap_aligns_ids_and_identical_predictions_have_zero_gap():
    records=[{'id':str(i),'true_label':str(i%2),'predicted_label':str(i%2)} for i in range(20)]
    result=module.paired_interval(records,[records[::-1]],['0','1'],samples=100)
    assert result['difference']==0
    assert result['interval_95']==[0,0]
    with pytest.raises(ValueError,match='Unpaired'):
        module.paired_interval(records,[records[:-1]],['0','1'],samples=10)

def test_task_instruction_reaches_api():
    def handler(request):
        import json
        assert json.loads(request.content)['questions']['category']['instructions']=='Detect emotion.'
        return httpx.Response(200,json={'answers':{'category':{'type':'choice','choice':'joy','probabilities':{'joy':1.0},'confidence':1.0}}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result=classify(client,{'id':'a','text':'happy','label':'joy','split':'test'},['joy'],{'joy':'Happy'},
                        {'model':'test','instructions':'Detect emotion.','max_attempts':1})
    assert result['error'] is None
