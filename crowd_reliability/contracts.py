"""Canonical observation boundary shared by fitting and live inference."""
import math
from datetime import datetime, timezone
UNITS={'density':'p/m2','speed':'m/s','inflow':'p/s','outflow':'p/s','queue_length':'persons','exit_open':'binary'}
def field(record,key,default=None):
    return record.get(key,default) if isinstance(record,dict) else getattr(record,key,default)
def utc(value):
    if isinstance(value,str):value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if not isinstance(value,datetime):raise ValueError('Expected an ISO timestamp.')
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value

def select_reports(observations,as_of,scenario_id=None,zone_id=None):
    cutoff=utc(as_of); selected={}; identities={}
    for o in observations:
        if scenario_id is not None and field(o,'scenario_id') not in (None,scenario_id):continue
        if zone_id is not None and field(o,'zone_id')!=zone_id:continue
        received=field(o,'received_at')
        if received is None or utc(received)>cutoff:continue
        variable=field(o,'variable')
        if variable not in UNITS:continue
        observed=utc(field(o,'observed_at'));received=utc(received)
        published=utc(field(o,'published_at',received) or received)
        if not observed<=published<=received:raise ValueError('Require observed_at <= published_at <= received_at.')
        unit=field(o,'unit')
        if unit is not None and unit!=UNITS[variable]:raise ValueError('Incompatible unit for '+variable)
        value=field(o,'value')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Observation must be finite and nonnegative.')
        if variable=='exit_open' and value not in (0,1):raise ValueError('Observed exit status must be 0 or 1.')
        rid=field(o,'report_id')
        signature=(field(o,'scenario_id'),field(o,'zone_id'),variable,field(o,'source_id'),observed,received,value,unit)
        if rid:
            if rid in identities and identities[rid]!=signature:raise ValueError('Conflicting reuse of report_id.')
            identities[rid]=signature
        key=(field(o,'zone_id'),variable)
        rank=(observed,received,str(rid or ''),str(field(o,'source_id','')),float(value))
        if key not in selected or rank>selected[key][0]:selected[key]=(rank,o)
    return [selected[k][1] for k in sorted(selected)]
