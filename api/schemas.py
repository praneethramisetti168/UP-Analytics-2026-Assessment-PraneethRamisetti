"""
APS Failure Prediction API — Schemas
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SensorInput(BaseModel):
    """
    Input schema for a single truck sensor reading.
    All 170 sensor fields are optional (mirrors real-world missing data).
    The API will handle missing values using the same median imputation
    strategy used during model training.
    """
    # Core sensor fields (abbreviated for readability; all 170 accepted)
    aa_000: Optional[float] = None
    ab_000: Optional[float] = None
    ac_000: Optional[float] = None
    ad_000: Optional[float] = None
    ae_000: Optional[float] = None
    af_000: Optional[float] = None
    ag_000: Optional[float] = None
    ag_001: Optional[float] = None
    ag_002: Optional[float] = None
    ag_003: Optional[float] = None
    ag_004: Optional[float] = None
    ag_005: Optional[float] = None
    ag_006: Optional[float] = None
    ag_007: Optional[float] = None
    ag_008: Optional[float] = None
    ag_009: Optional[float] = None
    ah_000: Optional[float] = None
    ai_000: Optional[float] = None
    aj_000: Optional[float] = None
    ak_000: Optional[float] = None
    al_000: Optional[float] = None
    am_0: Optional[float] = None
    an_000: Optional[float] = None
    ao_000: Optional[float] = None
    ap_000: Optional[float] = None
    aq_000: Optional[float] = None
    ar_000: Optional[float] = None
    as_000: Optional[float] = None
    at_000: Optional[float] = None
    au_000: Optional[float] = None
    av_000: Optional[float] = None
    ax_000: Optional[float] = None
    ay_000: Optional[float] = None
    ay_001: Optional[float] = None
    ay_002: Optional[float] = None
    ay_003: Optional[float] = None
    ay_004: Optional[float] = None
    ay_005: Optional[float] = None
    ay_006: Optional[float] = None
    ay_007: Optional[float] = None
    ay_008: Optional[float] = None
    ay_009: Optional[float] = None
    az_000: Optional[float] = None
    az_001: Optional[float] = None
    az_002: Optional[float] = None
    az_003: Optional[float] = None
    az_004: Optional[float] = None
    az_005: Optional[float] = None
    az_006: Optional[float] = None
    az_007: Optional[float] = None
    az_008: Optional[float] = None
    az_009: Optional[float] = None
    ba_000: Optional[float] = None
    ba_001: Optional[float] = None
    ba_002: Optional[float] = None
    ba_003: Optional[float] = None
    ba_004: Optional[float] = None
    ba_005: Optional[float] = None
    ba_006: Optional[float] = None
    ba_007: Optional[float] = None
    ba_008: Optional[float] = None
    ba_009: Optional[float] = None
    bb_000: Optional[float] = None
    bc_000: Optional[float] = None
    bd_000: Optional[float] = None
    be_000: Optional[float] = None
    bf_000: Optional[float] = None
    bg_000: Optional[float] = None
    bh_000: Optional[float] = None
    bi_000: Optional[float] = None
    bj_000: Optional[float] = None
    bk_000: Optional[float] = None
    bl_000: Optional[float] = None
    bm_000: Optional[float] = None
    bn_000: Optional[float] = None
    bo_000: Optional[float] = None
    bp_000: Optional[float] = None
    bq_000: Optional[float] = None
    br_000: Optional[float] = None
    bs_000: Optional[float] = None
    bt_000: Optional[float] = None
    bu_000: Optional[float] = None
    bv_000: Optional[float] = None
    bx_000: Optional[float] = None
    by_000: Optional[float] = None
    bz_000: Optional[float] = None
    ca_000: Optional[float] = None
    cb_000: Optional[float] = None
    cc_000: Optional[float] = None
    cd_000: Optional[float] = None
    ce_000: Optional[float] = None
    cf_000: Optional[float] = None
    cg_000: Optional[float] = None
    ch_000: Optional[float] = None
    ci_000: Optional[float] = None
    cj_000: Optional[float] = None
    ck_000: Optional[float] = None
    cl_000: Optional[float] = None
    cm_000: Optional[float] = None
    cn_000: Optional[float] = None
    cn_001: Optional[float] = None
    cn_002: Optional[float] = None
    cn_003: Optional[float] = None
    cn_004: Optional[float] = None
    cn_005: Optional[float] = None
    cn_006: Optional[float] = None
    cn_007: Optional[float] = None
    cn_008: Optional[float] = None
    cn_009: Optional[float] = None
    co_000: Optional[float] = None
    cp_000: Optional[float] = None
    cq_000: Optional[float] = None
    cs_000: Optional[float] = None
    cs_001: Optional[float] = None
    cs_002: Optional[float] = None
    cs_003: Optional[float] = None
    cs_004: Optional[float] = None
    cs_005: Optional[float] = None
    cs_006: Optional[float] = None
    cs_007: Optional[float] = None
    cs_008: Optional[float] = None
    cs_009: Optional[float] = None
    ct_000: Optional[float] = None
    cu_000: Optional[float] = None
    cv_000: Optional[float] = None
    cx_000: Optional[float] = None
    cy_000: Optional[float] = None
    cz_000: Optional[float] = None
    da_000: Optional[float] = None
    db_000: Optional[float] = None
    dc_000: Optional[float] = None
    dd_000: Optional[float] = None
    de_000: Optional[float] = None
    df_000: Optional[float] = None
    dg_000: Optional[float] = None
    dh_000: Optional[float] = None
    di_000: Optional[float] = None
    dj_000: Optional[float] = None
    dk_000: Optional[float] = None
    dl_000: Optional[float] = None
    dm_000: Optional[float] = None
    dn_000: Optional[float] = None
    do_000: Optional[float] = None
    dp_000: Optional[float] = None
    dq_000: Optional[float] = None
    dr_000: Optional[float] = None
    ds_000: Optional[float] = None
    dt_000: Optional[float] = None
    du_000: Optional[float] = None
    dv_000: Optional[float] = None
    dx_000: Optional[float] = None
    dy_000: Optional[float] = None
    dz_000: Optional[float] = None
    ea_000: Optional[float] = None
    eb_000: Optional[float] = None
    ec_00: Optional[float] = None
    ed_000: Optional[float] = None
    ee_000: Optional[float] = None
    ee_001: Optional[float] = None
    ee_002: Optional[float] = None
    ee_003: Optional[float] = None
    ee_004: Optional[float] = None
    ee_005: Optional[float] = None
    ee_006: Optional[float] = None
    ee_007: Optional[float] = None
    ee_008: Optional[float] = None
    ee_009: Optional[float] = None
    ef_000: Optional[float] = None
    eg_000: Optional[float] = None

    class Config:
        extra = "allow"   # Accept any additional sensor fields not listed above


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="0=Non-APS failure, 1=APS failure")
    predicted_label: str = Field(..., description="Human-readable label")
    probability: float = Field(..., description="APS failure probability (0-1)")
    risk_bucket: str = Field(..., description="High / Medium / Low")
    recommendation: str = Field(..., description="Operational recommendation")
    confidence: str = Field(..., description="Model confidence level")
    threshold_used: float = Field(..., description="Decision threshold applied")


class BatchSensorInput(BaseModel):
    records: List[SensorInput] = Field(..., description="List of truck sensor readings")


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    test_pr_auc: float
    test_roc_auc: float
    threshold: float
    top_features: List[str]
    endpoints: List[str]
