import numpy as np


def to_dict(df):
    out = dict()

    for i, row in df.iterrows():
        out[row['EQP_NM']] = {
            'X': row['X'],
            'Y': row['Y'],
            'WARP': row['WARP']
        }

    return out


def get_golden_tool(out, eqp_nm):
    refer = out[eqp_nm]
    x0 = refer['X']
    y0 = refer['Y']

    eqp_nms = []
    scores = []
    for k, data in out.items():
        eqp_nms.append(k)

        x1 = data['X']
        y1 = data['Y']

        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        warp = data['WARP']

        score = dist * warp
        scores.append(score)

    idx = np.argsort(scores)
    eqp_nms = np.array(eqp_nms)[idx]

    return list(filter(lambda x: x != eqp_nm, eqp_nms))
