import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
import io
import time
from plotly.subplots import make_subplots
from openai import OpenAI
import json

# OpenAI APIクライアントの初期化
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ユーザーのOpenAI APIキーを設定
openai_api_key = st.text_input("OpenAI APIキーを入力してください", type="password")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.warning("⚠️ OpenAI APIキーを入力してください。APIキーがないとAI分析機能は使用できません。")

def analyze_stl_results(daily_data, res, type_selected):
    """STL分析結果を分析する関数"""
    
    # APIキーが設定されていない場合は早期リターン
    if not openai_api_key:
        return "OpenAI APIキーが設定されていません。APIキーを入力してからもう一度お試しください。"
    
    # 時系列の特徴を抽出
    trend = res.trend
    trend_diff = trend.diff()  # トレンドの変化率
    
    # 変化点の検出（トレンドの傾きが変化する点）
    change_points = []
    trend_diff_values = trend_diff.values[1:]  # 最初のNaNを除外
    trend_diff_sign = np.sign(trend_diff_values)
    sign_changes = np.where(np.diff(trend_diff_sign) != 0)[0]
    
    # インデックスを1つずらして変化点を取得（diffによるずれを補正）
    for idx in sign_changes:
        actual_idx = idx + 1  # diffによるずれを補正
        if actual_idx < len(trend_diff):
            change_points.append({
                "date": trend_diff.index[actual_idx].strftime("%Y-%m-%d"),
                "value": float(trend.iloc[actual_idx]),
                "direction": "上昇→下降" if trend_diff.iloc[actual_idx] < 0 else "下降→上昇"
            })
    
    # 期間ごとの統計
    total_days = (daily_data.index[-1] - daily_data.index[0]).days
    start_date = daily_data.index[0].strftime("%Y-%m-%d")
    end_date = daily_data.index[-1].strftime("%Y-%m-%d")
    
    # トレンドの全体的な変化
    total_change = float(trend.iloc[-1] - trend.iloc[0])
    avg_daily_change = float(total_change / total_days) if total_days > 0 else 0
    
    # 最大の変化率を記録した期間
    max_increase_idx = trend_diff.idxmax()
    max_decrease_idx = trend_diff.idxmin()
    
    # 季節性の強さ（振幅）の時間変化
    seasonal_amplitude = pd.Series(np.abs(res.seasonal)).resample('W').mean()
    max_seasonal_week = seasonal_amplitude.idxmax().strftime("%Y-%m-%d")
    min_seasonal_week = seasonal_amplitude.idxmin().strftime("%Y-%m-%d")
    
    # 統計情報の作成
    stats = {
        "基本統計": {
            "original": {
                "mean": float(daily_data.mean()),
                "std": float(daily_data.std()),
                "min": float(daily_data.min()),
                "max": float(daily_data.max())
            },
            "trend": {
                "mean": float(trend.mean()),
                "std": float(trend.std()),
                "min": float(trend.min()),
                "max": float(trend.max())
            },
            "seasonal": {
                "mean": float(res.seasonal.mean()),
                "std": float(res.seasonal.std()),
                "min": float(res.seasonal.min()),
                "max": float(res.seasonal.max())
            },
            "residual": {
                "mean": float(res.resid.mean()),
                "std": float(res.resid.std()),
                "min": float(res.resid.min()),
                "max": float(res.resid.max())
            }
        },
        "時系列特徴": {
            "観測期間": {
                "開始日": start_date,
                "終了日": end_date,
                "総日数": total_days
            },
            "トレンド変化": {
                "総変化量": total_change,
                "1日あたりの平均変化": avg_daily_change,
                "最大増加日": max_increase_idx.strftime("%Y-%m-%d"),
                "最大減少日": max_decrease_idx.strftime("%Y-%m-%d")
            },
            "変化点": change_points,
            "季節性の特徴": {
                "最も強い週": max_seasonal_week,
                "最も弱い週": min_seasonal_week,
                "平均振幅": float(seasonal_amplitude.mean())
            }
        }
    }
    
    # 分析用のプロンプトを作成
    prompt = f"""
    以下のSTL分析結果を、専門家の視点から詳細に分析してください。データタイプは{type_selected}です。これらのデータは1個人のヘルスケアデータに基づきます。

    提供情報：
    {json.dumps(stats, indent=2, ensure_ascii=False)}

    以下の観点に基づいて、段落ごとに分かりやすく日本語で解説してください。
    
    【分析観点】
    1. 元データの分析：
    - 全体的な傾向や特徴を読み取り、平均値、変動幅、ばらつきについてまとめてください。
    - 観測期間全体（{start_date}から{end_date}まで）での変化の概要を説明してください。

    2. トレンド成分の分析：
    - 長期的な上昇・下降傾向について詳しく解釈してください。
    - 特に検出された変化点（{len(change_points)}箇所）について、その時期と変化の性質を具体的に説明してください。
    - 最大の増加（{max_increase_idx.strftime("%Y-%m-%d")}）と最大の減少（{max_decrease_idx.strftime("%Y-%m-%d")}）が見られた時期とその意味について考察してください。

    3. 季節性成分の分析：
    - 周期的なパターンが見られるか、またその大きさと周期の特徴を説明してください。
    - 特に季節性が強かった期間（{max_seasonal_week}周辺）と弱かった期間（{min_seasonal_week}周辺）について考察してください。

    4. 残差成分の分析：
    - 残差の特徴（ランダム性、例外的な変動など）を指摘してください。
    - 特に大きな残差が見られる場合、その時期と考えられる要因について考察してください。

    5. 総合的な考察と示唆：
    - データ全体から読み取れる健康管理上のストーリーを時系列で説明してください。
    - 1日あたりの平均変化（{avg_daily_change:.3f}）は適切な範囲といえるか、評価してください。
    - 検出された変化点は、生活習慣の変化やイベントと関連している可能性について考察してください。
    - 今後の健康管理への具体的な提案をしてください。

    ※各セクションは論理的かつ自然な日本語でまとめ、専門知識がない人にも伝わるようなわかりやすい説明を心がけてください。
    """

    # デバッグ用の表示
#    with st.expander("🔍 デバッグ: 分析用データとプロンプト"):
#        st.write("### 統計情報")
#        st.json(stats)
#        st.write("### 生成プロンプト")
#        st.code(prompt, language="text")

    try:
        # OpenAI APIを使用して分析を実行
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "あなたはデータ分析の専門家です。STL分析の結果を詳細に分析し、分かりやすく説明してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"分析中にエラーが発生しました: {str(e)}"

st.title("📊 ヘルスケアデータ可視化アプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")

if uploaded_file is not None:
    # セッションステートの初期化
    if "df" not in st.session_state:
        progress_text = st.empty()
        progress_text.write("CSVファイルを読み込み中...")
        df = pd.read_csv(uploaded_file)
        
        progress_text.write("タイムゾーンを変換中...")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")

        # 2024年4月以降に絞る
      #  start_filter = pd.Timestamp("2024-04-01", tz="Asia/Tokyo")
      #  df = df[df["datetime"] >= start_filter]
        progress_text.write("✅ データの読み込みが完了しました！")
        
        st.session_state.df = df
        st.session_state.progress_text = progress_text
    else:
        df = st.session_state.df
        progress_text = st.session_state.progress_text

    # typeの一覧を取得
    types = df["type"].unique()

    # サイドバーで選択肢を設定
    st.sidebar.header("📊 パラメータ設定")
    
    # セッションステートの初期化
    if "current_type" not in st.session_state:
        st.session_state.current_type = types[0]
        st.session_state.current_date_range = [df["datetime"].min().date(), df["datetime"].max().date()]
        st.session_state.current_period = 7
        st.session_state.filtered_data = None
        st.session_state.daily_data = None
        st.session_state.stl_result = None
        st.session_state.chat_history = []

    # パラメータ入力（ただし直接は更新しない）
    type_selected = st.sidebar.selectbox(
        "表示するデータの種類",
        types,
        index=list(types).index(st.session_state.current_type)
    )
    date_range = st.sidebar.date_input(
        "観察期間を選択",
        st.session_state.current_date_range
    )
    period = st.sidebar.number_input(
        "STL周期 (日)",
        min_value=2,
        max_value=60,
        value=st.session_state.current_period
    )

    # 解析ボタン
    if st.sidebar.button("🔄 グラフを描画"):
        progress_text.write("データをフィルタリング中...")
        
        # パラメータの更新
        st.session_state.current_type = type_selected
        st.session_state.current_date_range = date_range
        st.session_state.current_period = period
        
        # データのフィルタリングと計算
        start_date = pd.Timestamp(date_range[0], tz="Asia/Tokyo")
        end_date = pd.Timestamp(date_range[1], tz="Asia/Tokyo")
        mask = (
            (df["type"] == type_selected) &
            (df["datetime"] >= start_date) &
            (df["datetime"] <= end_date)
        )
        st.session_state.filtered_data = df[mask].copy()
        
        try:
            progress_text.write("データを数値に変換中...")
            st.session_state.filtered_data["value"] = pd.to_numeric(st.session_state.filtered_data["value"], errors="coerce")
            st.session_state.filtered_data.set_index("datetime", inplace=True)
            st.session_state.daily_data = st.session_state.filtered_data["value"].resample("D").mean().interpolate()
            
            if st.session_state.daily_data.isna().all():
                st.warning("選択された期間にデータがありません。")
                st.session_state.filtered_data = None
                st.session_state.daily_data = None
                st.session_state.stl_result = None
            elif len(st.session_state.daily_data) >= period:
                progress_text.write("STL分解を実行中...")
                stl = STL(st.session_state.daily_data, period=period)
                st.session_state.stl_result = stl.fit()
                progress_text.write("✅ データの処理が完了しました！")
            else:
                st.warning("データ数が少なすぎてSTL分解できません。もう少し長めの観察期間を選んでください。")
                st.session_state.stl_result = None
        except ValueError:
            st.warning("データの形式が正しくありません。数値データのみを処理できます。")
            st.session_state.filtered_data = None
            st.session_state.daily_data = None
            st.session_state.stl_result = None
        
        # チャット履歴をクリア
        st.session_state.chat_history = []

    # 保存されているデータとSTL結果を使用してグラフを描画
    if st.session_state.stl_result is not None:
        st.subheader(f"📈 STL分解グラフ：{st.session_state.current_type}")
        st.caption(f"観測期間：{st.session_state.current_date_range[0]} ～ {st.session_state.current_date_range[1]}")
        st.caption(f"周期：{st.session_state.current_period}日")

        # 基本統計量の表示
        st.subheader("📊 基本統計量")
        stats_df = pd.DataFrame({
            "平均値": [
                float(st.session_state.daily_data.mean()),
                float(st.session_state.stl_result.trend.mean()),
                float(st.session_state.stl_result.seasonal.mean()),
                float(st.session_state.stl_result.resid.mean())
            ],
            "最大値": [
                float(st.session_state.daily_data.max()),
                float(st.session_state.stl_result.trend.max()),
                float(st.session_state.stl_result.seasonal.max()),
                float(st.session_state.stl_result.resid.max())
            ],
            "最小値": [
                float(st.session_state.daily_data.min()),
                float(st.session_state.stl_result.trend.min()),
                float(st.session_state.stl_result.seasonal.min()),
                float(st.session_state.stl_result.resid.min())
            ],
            "標準偏差": [
                float(st.session_state.daily_data.std()),
                float(st.session_state.stl_result.trend.std()),
                float(st.session_state.stl_result.seasonal.std()),
                float(st.session_state.stl_result.resid.std())
            ],
            "データ数": [
                len(st.session_state.daily_data),
                len(st.session_state.stl_result.trend),
                len(st.session_state.stl_result.seasonal),
                len(st.session_state.stl_result.resid)
            ]
        }, index=["元データ", "トレンド成分", "季節性成分", "残差成分"])
        
        # 小数点以下2桁に丸める（データ数以外）
        for col in ["平均値", "最大値", "最小値", "標準偏差"]:
            stats_df[col] = stats_df[col].round(2)
        
        # テーブルとして表示
        st.table(stats_df)

        # 4つのサブプロットを作成
        fig = go.Figure()
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                "元データ (Original)",
                "トレンド成分 (Trend)",
                "季節性成分 (Seasonal)",
                "残差 (Residual)"
            ),
            vertical_spacing=0.1
        )

        # 元データ
        fig.add_trace(
            go.Scatter(x=st.session_state.daily_data.index, y=st.session_state.daily_data, name="Original", line=dict(color="#1f77b4")),
            row=1, col=1
        )

        # トレンド成分
        fig.add_trace(
            go.Scatter(x=st.session_state.daily_data.index, y=st.session_state.stl_result.trend, name="Trend", line=dict(color="#2ca02c")),
            row=2, col=1
        )

        # 季節性成分
        fig.add_trace(
            go.Scatter(x=st.session_state.daily_data.index, y=st.session_state.stl_result.seasonal, name="Seasonal", line=dict(color="#ff7f0e")),
            row=3, col=1
        )

        # 残差
        fig.add_trace(
            go.Scatter(x=st.session_state.daily_data.index, y=st.session_state.stl_result.resid, name="Residual", line=dict(color="#d62728")),
            row=4, col=1
        )

        # レイアウトの調整
        fig.update_layout(
            height=800,
            title=f"STL分解 - {st.session_state.current_type}",
            showlegend=False,
            xaxis4_title="日付",
            font=dict(
                family="IPAexGothic",
                size=12
            )
        )

        # Y軸ラベルを追加
        fig.update_yaxes(title_text="値", row=1, col=1)
        fig.update_yaxes(title_text="トレンド", row=2, col=1)
        fig.update_yaxes(title_text="季節性", row=3, col=1)
        fig.update_yaxes(title_text="残差", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # 解析とチャットのタブを作成
        analysis_tab, chat_tab = st.tabs(["📊 STL分析の解析", "💬 グラフについての質問"])

        with analysis_tab:
            if st.button("🔍 STL分析を解析する"):
                st.subheader("📊 STL分析結果の解釈")
                with st.spinner("データを分析中..."):
                    analysis_result = analyze_stl_results(
                        st.session_state.daily_data,
                        st.session_state.stl_result,
                        st.session_state.current_type
                    )
                    st.markdown(analysis_result)

        with chat_tab:
            # チャット履歴の表示
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # ユーザーの入力欄
            if prompt := st.chat_input("グラフについて質問してください"):
                # ユーザーのメッセージを表示
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 統計情報の準備
                stats = {
                    "original": {
                        "mean": float(st.session_state.daily_data.mean()),
                        "std": float(st.session_state.daily_data.std()),
                        "min": float(st.session_state.daily_data.min()),
                        "max": float(st.session_state.daily_data.max())
                    },
                    "trend": {
                        "mean": float(st.session_state.stl_result.trend.mean()),
                        "std": float(st.session_state.stl_result.trend.std()),
                        "min": float(st.session_state.stl_result.trend.min()),
                        "max": float(st.session_state.stl_result.trend.max())
                    },
                    "seasonal": {
                        "mean": float(st.session_state.stl_result.seasonal.mean()),
                        "std": float(st.session_state.stl_result.seasonal.std()),
                        "min": float(st.session_state.stl_result.seasonal.min()),
                        "max": float(st.session_state.stl_result.seasonal.max())
                    },
                    "residual": {
                        "mean": float(st.session_state.stl_result.resid.mean()),
                        "std": float(st.session_state.stl_result.resid.std()),
                        "min": float(st.session_state.stl_result.resid.min()),
                        "max": float(st.session_state.stl_result.resid.max())
                    }
                }

                # アシスタントの応答を生成
                with st.chat_message("assistant"):
                    with st.spinner("回答を生成中..."):
                        context = f"""
                        現在表示されているのは{st.session_state.current_type}のSTL分解グラフです。
                        期間は{st.session_state.current_date_range[0]}から{st.session_state.current_date_range[1]}まで、周期は{st.session_state.current_period}日です。

                        データの基本統計量:
                        {json.dumps(stats, indent=2, ensure_ascii=False)}

                        ユーザーからの質問: {prompt}

                        グラフの特徴を踏まえて、専門的かつ分かりやすく回答してください。
                        """

                        response = client.chat.completions.create(
                            model="gpt-4.1-nano",
                            messages=[
                                {"role": "system", "content": "あなたはデータ分析の専門家です。STL分解グラフについての質問に答えてください。"},
                                {"role": "user", "content": context}
                            ],
                            temperature=0.7,
                            max_tokens=2000
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)

                # チャット履歴の更新
                st.session_state.chat_history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ])

        # PNGグラフの生成と保存
        progress_text.write("PNGグラフを生成中...")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 日本語フォントの設定
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'IPAGothic', 'Yu Gothic', 'Meiryo', 'MS Gothic']

        # グラフの色を定義
        colors = {
            'original': '#1f77b4',    # 青
            'trend': '#2ca02c',       # 緑
            'seasonal': '#ff7f0e',    # オレンジ
            'residual': '#d62728'     # 赤
        }

        # メインタイトルとサブタイトルの作成
        main_title = f"STL分解 - {st.session_state.current_type}"
        sub_title = f"観測期間：{st.session_state.current_date_range[0]} ～ {st.session_state.current_date_range[1]}  |  周期：{st.session_state.current_period}日"

        fig_png, axs = plt.subplots(4, 1, figsize=(12, 10))
        
        # メインタイトルとサブタイトルを追加
        fig_png.suptitle(main_title, fontsize=14, y=0.95)
        # サブタイトルを追加（yパラメータで位置を調整）
        fig_png.text(0.5, 0.91, sub_title, fontsize=10, ha='center')

        # 各グラフをPlotlyと同じ色で描画
        axs[0].plot(st.session_state.daily_data.index, st.session_state.daily_data, color=colors['original'], linewidth=2)
        axs[0].set_title("元のデータ (Original)", fontsize=10, pad=10)
        axs[1].plot(st.session_state.daily_data.index, st.session_state.stl_result.trend, color=colors['trend'], linewidth=2)
        axs[1].set_title("トレンド成分 (Trend)", fontsize=10, pad=10)
        axs[2].plot(st.session_state.daily_data.index, st.session_state.stl_result.seasonal, color=colors['seasonal'], linewidth=2)
        axs[2].set_title("季節性成分 (Seasonal)", fontsize=10, pad=10)
        axs[3].plot(st.session_state.daily_data.index, st.session_state.stl_result.resid, color=colors['residual'], linewidth=2)
        axs[3].set_title("残差 (Residual)", fontsize=10, pad=10)
        
        # X軸とY軸のラベルを設定
        axs[3].set_xlabel("日付", fontsize=10)
        for ax in axs:
            ax.set_ylabel("値", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # レイアウトの調整（タイトル用のスペースを確保）
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        # バイナリで保存しダウンロードボタン表示
        buf = io.BytesIO()
        fig_png.savefig(buf, format="png")
        st.download_button("📥 グラフをPNGでダウンロード", data=buf.getvalue(), file_name="stl_plot.png", mime="image/png")
        
        progress_text.write("✅ グラフの生成が完了しました！")

else:
    st.info("CSVファイルをアップロードしてください。")
