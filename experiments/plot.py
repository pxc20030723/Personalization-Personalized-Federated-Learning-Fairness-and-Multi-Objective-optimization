import os
import pandas as pd
import matplotlib.pyplot as plt

# 保证中文环境也能正常显示负号
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ====== 各实验结果 CSV 路径 ======
FEDAVG_CSV = os.path.join(BASE_DIR, 'results', 'training_history_fedavg.csv')
DITTO_CSV = os.path.join(BASE_DIR, 'results', 'ditto', 'training_history.csv')
QFFL_CSV = os.path.join(BASE_DIR, 'results', 'qffl', 'training_history.csv')
MGDA_CSV = os.path.join(BASE_DIR, 'results', 'mgda', 'training_history.csv')
PERFL_CSV = os.path.join(BASE_DIR, 'results', 'perFL', 'training_history.csv')

PLOT_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


def load_histories():
    """读取所有需要的 CSV"""
    hist = {}
    hist['fedavg'] = pd.read_csv(FEDAVG_CSV)
    hist['ditto'] = pd.read_csv(DITTO_CSV)
    hist['qffl'] = pd.read_csv(QFFL_CSV)
    hist['mgda'] = pd.read_csv(MGDA_CSV)
    hist['perfl'] = pd.read_csv(PERFL_CSV)
    return hist


def plot_overall_loss_rmse(hist):
    """两张总览图：loss 和 RMSE，五条线（FedAvg、Ditto、qFFL、MGDA、perFL）"""

    # ====== 1) Loss 图 ======
    plt.figure(figsize=(8, 5))

    # 横轴：communication rounds
    rounds_fed = hist['fedavg']['round']
    rounds_other = hist['ditto']['round']  # 其它几个 round 一致

    # Loss 选取：
    # - FedAvg: avg_train_loss
    # - Ditto, qFFL, MGDA, perFL: avg_personalized_train_loss
    plt.plot(rounds_fed, hist['fedavg']['avg_train_loss'],
             label='FedAvg', linewidth=2)

    plt.plot(rounds_other, hist['ditto']['avg_personalized_train_loss'],
             label='Ditto', linewidth=2)

    plt.plot(rounds_other, hist['qffl']['avg_personalized_train_loss'],
             label='q-FFL', linewidth=2)

    plt.plot(rounds_other, hist['mgda']['avg_personalized_train_loss'],
             label='MGDA', linewidth=2)

    plt.plot(rounds_other, hist['perfl']['avg_personalized_train_loss'],
             label='perFL (q-FFL+MGDA)', linewidth=2)

    plt.xlabel('Communication Rounds')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Communication Rounds')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(PLOT_DIR, 'overall_loss.png')
    plt.savefig(loss_path, dpi=300)
    print(f'Saved: {loss_path}')

    # ====== 2) RMSE 图 ======
    plt.figure(figsize=(8, 5))

    # RMSE 选取：
    # - FedAvg: avg_test_rmse
    # - Ditto, qFFL, MGDA, perFL: avg_personalized_test_rmse
    plt.plot(rounds_fed, hist['fedavg']['avg_test_rmse'],
             label='FedAvg', linewidth=2)

    plt.plot(rounds_other, hist['ditto']['avg_personalized_test_rmse'],
             label='Ditto', linewidth=2)

    plt.plot(rounds_other, hist['qffl']['avg_personalized_test_rmse'],
             label='q-FFL', linewidth=2)

    plt.plot(rounds_other, hist['mgda']['avg_personalized_test_rmse'],
             label='MGDA', linewidth=2)

    plt.plot(rounds_other, hist['perfl']['avg_personalized_test_rmse'],
             label='perFL (q-FFL+MGDA)', linewidth=2)

    plt.xlabel('Communication Rounds')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs Communication Rounds')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    rmse_path = os.path.join(PLOT_DIR, 'overall_rmse.png')
    plt.savefig(rmse_path, dpi=300)
    print(f'Saved: {rmse_path}')


def plot_pair_with_ditto(hist):
    """四张图：FedAvg、qFFL、MGDA、perFL 各自与 Ditto 的 RMSE 对比"""

    rounds_ditto = hist['ditto']['round']
    ditto_rmse = hist['ditto']['avg_personalized_test_rmse']

    # 1) FedAvg vs Ditto
    plt.figure(figsize=(8, 5))
    plt.plot(hist['fedavg']['round'], hist['fedavg']['avg_test_rmse'],
             label='FedAvg', linewidth=2)
    plt.plot(rounds_ditto, ditto_rmse,
             label='Ditto (personalized)', linewidth=2)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test RMSE')
    plt.title('FedAvg vs Ditto (Test RMSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'fedavg_vs_ditto_rmse.png')
    plt.savefig(path, dpi=300)
    print(f'Saved: {path}')

    # 2) q-FFL vs Ditto
    plt.figure(figsize=(8, 5))
    plt.plot(rounds_ditto, hist['qffl']['avg_personalized_test_rmse'],
             label='q-FFL (personalized)', linewidth=2)
    plt.plot(rounds_ditto, ditto_rmse,
             label='Ditto (personalized)', linewidth=2)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test RMSE')
    plt.title('q-FFL vs Ditto (Test RMSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'qffl_vs_ditto_rmse.png')
    plt.savefig(path, dpi=300)
    print(f'Saved: {path}')

    # 3) MGDA vs Ditto
    plt.figure(figsize=(8, 5))
    plt.plot(rounds_ditto, hist['mgda']['avg_personalized_test_rmse'],
             label='MGDA (personalized)', linewidth=2)
    plt.plot(rounds_ditto, ditto_rmse,
             label='Ditto (personalized)', linewidth=2)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test RMSE')
    plt.title('MGDA vs Ditto (Test RMSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'mgda_vs_ditto_rmse.png')
    plt.savefig(path, dpi=300)
    print(f'Saved: {path}')

    # 4) perFL(q-FFL+MGDA) vs Ditto
    plt.figure(figsize=(8, 5))
    plt.plot(rounds_ditto, hist['perfl']['avg_personalized_test_rmse'],
             label='perFL (q-FFL+MGDA, personalized)', linewidth=2)
    plt.plot(rounds_ditto, ditto_rmse,
             label='Ditto (personalized)', linewidth=2)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test RMSE')
    plt.title('perFL (q-FFL+MGDA) vs Ditto (Test RMSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'perfl_vs_ditto_rmse.png')
    plt.savefig(path, dpi=300)
    print(f'Saved: {path}')


def main():
    hist = load_histories()
    plot_overall_loss_rmse(hist)
    plot_pair_with_ditto(hist)
    print('All plots generated.')


if __name__ == '__main__':
    main()