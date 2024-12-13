from flask import Flask, render_template, request, jsonify
import os
import h5py
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import plotly.colors

# グローバル変数としてh5_files、tracer_list、max_tracerを定義
h5_files = []
tracer_list = []
max_tracer = 20

# 閾値計算とstart_indexの算出
def extract_segment(load_data, threshold_coefficient, backward_points):
    threshold = load_data[0] * threshold_coefficient
    start_index = np.argmax(load_data >= threshold)
    start_index = max(0, start_index - backward_points)

    offset = abs(load_data[start_index] - load_data[0])
    load_data_offset = load_data - offset
    load_data_offset = np.maximum(load_data_offset, 0)
    
    threshold = threshold_coefficient / 1.8
    start_index = np.argmax(load_data_offset >= threshold)
    start_index = max(0, start_index - backward_points)
    return start_index

app = Flask(__name__)

def initialize_h5_files():
    global h5_files
    data_folder = 'data'
    threshold_coefficient = 300
    backward_points = 75
    sample_num = 225

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.h5'):
                full_path = os.path.join(root, file)
                with h5py.File(full_path, 'r') as f:
                    sampling_freq = f['Measure/SamplingFreq'][()]
                    time_step = 1.0 / sampling_freq
                    sort_list = []

                    # PhyCoefとRangeCoefを抽出
                    phy_coef = f['Measure/Ch/PhyCoef'][()]
                    range_coef = f['Measure/Ch/RangeCoef'][()]

                    for index in f['Shot'].keys():
                        if index.isdigit():
                            shot_num = int(f[f'Shot/{index}/ShotNum'][()][0])
                            raw_data = f[f'Shot/{index}/RawData'][()]
                            load_data = raw_data[:4, :].T

                            # PhyCoefとRangeCoefをload_dataに適用
                            scaled_load_data = load_data * (phy_coef[:4] * range_coef[:4])

                            load1 = scaled_load_data[:, 0]

                            start_index = extract_segment(load1, threshold_coefficient, backward_points)
                            end_index = start_index + sample_num
                            load1_seg = load1[start_index:end_index]

                            sort_list.append({
                                'file_path': full_path,
                                'shot_num': shot_num,
                                'load1_seg': load1_seg,
                                'time_step': time_step,
                                'start_index': start_index
                            })

                    sort_list.sort(key=lambda x: x['shot_num'])
                    h5_files.extend(sort_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_file', methods=['POST'])
def load_file():
    global h5_files, tracer_list, max_tracer
    file_index = request.json.get('file_index', 0)
    
    if file_index < len(h5_files):
        file_path = h5_files[file_index]['file_path']
        shot_num = h5_files[file_index]['shot_num']
        load1_seg = h5_files[file_index]['load1_seg']
        time_step = h5_files[file_index]['time_step']
        start_index = h5_files[file_index]['start_index']

        # start_indexを利用して統一されたX軸を作成
        start_time = start_index * time_step * 1000  # start_indexに基づくスタート時間（ミリ秒）
        time_data = start_time + np.arange(len(load1_seg)) * time_step * 1000  # 統一されたX軸データ

        # load1_segとtime_dataをセットでtracer_listに追加
        if len(tracer_list) >= max_tracer:
            tracer_list.pop(0)
        tracer_list.append({'load1_seg': load1_seg, 'time_data': time_data})

        # 各インデックスのデータをまとめる
        data_matrix = np.array([tracer['load1_seg'] for tracer in tracer_list])
        # 各インデックスの中央値を計算し、標準波形とする
        standard_waveform = np.median(data_matrix, axis=0)
        # 二乗誤差を計算
        squared_errors = (load1_seg - standard_waveform) ** 2

        def set_graph_layout(fig, xaxis_title, yaxis_title, xaxis_range=None):
            fig.update_layout(
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                yaxis=dict(range=[-200, 1500], showgrid=True, gridcolor='lightgray', linecolor='black'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                    linecolor='black',
                    range=xaxis_range
                ),
                plot_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor='white',
                shapes=[
                    dict(
                        type='rect',
                        xref='paper',
                        yref='paper',
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(color='black', width=2)
                    )
                ]
            )

        # 測定波形グラフの作成
        fig = go.Figure()

        # Load1の波形を追加
        fig.add_trace(go.Scatter(
            x=time_data,
            y=standard_waveform,
            mode='lines',
            name='Standard Waveform',
            line=dict(color='lightgray', dash='dash')
        ))

        # Standard Waveformの波形を追加
        fig.add_trace(go.Scatter(
            x=time_data,
            y=load1_seg,
            mode='lines+markers',
            name='Load1',
            line=dict(color='blue'),
            marker=dict(
                size=5,
                color=squared_errors,
                colorscale='jet',
                colorbar=dict(
                    title='Error',
                    titleside='right',  # ラベルを横向きに表示
                    len=0.8,
                    yanchor='middle',
                    y=0.4,
                    x=1.05
                ),
                cmin=0,
                cmax=1.2e6
            )
        ))


        # グラフのレイアウトを設定
        set_graph_layout(fig, "time (ms)", "load force (N)", xaxis_range=[54, 56])

        graph_json = pio.to_json(fig)

        # トレーサーグラフの作成
        tracer_fig = go.Figure()
        for i, tracer in enumerate(tracer_list):
            tracer_time_data = tracer['time_data']
            tracer_data = tracer['load1_seg']
            color = f'rgb({int(255 * i / max_tracer)}, {int(255 * (max_tracer - i) / max_tracer)}, 0)'
            tracer_fig.add_trace(go.Scatter(x=tracer_time_data, y=tracer_data, mode='lines', name=f'Tracer {i+1}', line=dict(color=color)))

        # トレーサーグラフのレイアウトを設定
        set_graph_layout(tracer_fig, "time (ms)", "load force (N)", xaxis_range=[54, 56])

        tracer_graph_json = pio.to_json(tracer_fig)

        return jsonify({
            'file_path': file_path,
            'file_index': file_index,
            'shot_num': shot_num,
            'graph': graph_json,
            'tracer_graph': tracer_graph_json,
            'end_flag': False
        })
    else:
        return jsonify({'end_flag': True})

if __name__ == '__main__':
    initialize_h5_files()
    app.run(debug=True)