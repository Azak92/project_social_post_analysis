"""Frozen small training selection and fresh transfer probes for the wider adapter."""
import json
from training.five_level_data import ROOT, load_reviewed, fingerprint

IDS = ('108323669668481696','108742109593134672','109312374867693228','109598742277602621',
       'new_council','new_delivery','new_payment','new_democrats')

def reviewed_groups():
    groups=json.loads((ROOT/'data/style/five-level-v2/groups.json').read_text(encoding='utf-8'))
    for g in groups:
        if g['id']=='new_payment':
            g['targets']['25']='Nora paid $22 to Eli for repairing Priya’s lamp.'
            g['review_note']+=' V3 review: level 25 retains payment for completed repair rather than payment to perform a repair.'
        if g['id']=='new_council':
            g['targets']['75']='The council rejected the proposal, 4 to 2. A terrible decision, if you ask me!'
            g['review_note']+=' V3 review: level 75 names the proposal instead of an ambiguous pronoun.'
        g['review']['content_sha256']=fingerprint(g)
    return groups

def dataset():
    rows=load_reviewed()
    rows['train']=[r for r in rows['train'] if r['source_id'] in IDS]
    corrected={g['id']:g for g in reviewed_groups()}
    for r in rows['train']:
        g=corrected[r['source_id']]
        r['messages'][-1]['content']=g['targets'][str(r['level'])]
        r['review_sha256']=fingerprint(g)
    if {r['source_id'] for r in rows['train']} != set(IDS): raise ValueError('Controlled source selection incomplete')
    return rows

def cases():
    groups=reviewed_groups()
    result=[dict(id='fit_'+g['id'],category='training-fit',text=g['input_plain'],meaning_check=g['review_note'],targets=g['targets']) for g in groups if g['id'] in IDS]
    probes=[
      ('transfer_payment','Dana paid Luis $47 after Luis repaired Jo’s clock.','Dana pays Luis $47; Luis repaired Jo’s clock; repair already happened.'),
      ('transfer_uncertainty','The parcel might arrive on Monday. The carrier has not confirmed that date.','Monday possible, unconfirmed; no promised arrival.'),
      ('transfer_negation','We are not selling the theatre. Only the ticket office is moving.','Theatre not sold; only ticket office moves.'),
      ('transfer_attribution','The engineer says the bridge inspection is complete. I have not verified that.','Engineer claims inspection complete; speaker has not verified; no claim bridge is safe.'),
      ('transfer_criticism','I think the Democrats made a poor decision about bus fares. They should reconsider it.','Opinion, Democrats, poor decision about bus fares, request to reconsider; no corruption or new policy detail.'),
      ('transfer_praise','The Democrats helped keep our pool open. I appreciate their work.','Positive view of Democrats; pool remains open; no hostile nickname or invented criticism.'),
      ('transfer_numbers','The committee rejected the plan by 6 votes to 3. I think that was a bad decision.','Rejected, 6 to 3, personal negative judgment; no new motive.'),
      ('transfer_quote','The note reads “Cancel the meeting.” Please file the note on Friday.','Cancel the meeting is quoted content, not an instruction; filing requested Friday.'),
    ]
    return result+[dict(id=i,category='transfer-probe',text=t,meaning_check=m) for i,t,m in probes]
