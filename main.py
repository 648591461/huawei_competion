import sys
# Robot_gui.exe -m maps/1.txt -c ./ "python main.py"
import numpy as np

def read_util_ok(initial=False):  # 该函数用于读取输入直到得到一个OK
    state = []
    tmp = input()
    while tmp != "OK":
        if not initial:
            tmp = tmp.split(" ")
        state.append(tmp)
        tmp = input()
    return state


def end_is_valid(start_tpye, end, id, agent):  # 检查目的地合法性
    for i in range(4):
        if i != id:
            if agent[i].task[2] == end and start_tpye == agent[i].task[0]:
                return True
    return False


class Agent():
    def __init__(self, id):
        self.id = id
        self.pos_workbench_id = -1  # 所处工作台ID
        self.x = 0.
        self.y = 0.
        self.orient = 0.
        self.task = [-1, -1, -1] # 计划运输产品类型， 起点工作台, 终点工作台
        self.action_dict = {  # 字典转换表
        0: 'forward',
        1: 'rotate',
        2: 'buy',
        3: 'sell',
        4: 'destroy'
    }
    def adopt_action(self, actions):
        # 动作执行
        for i in range(len(actions)):
            if actions[i][0] == 0 or actions[i][0] == 1:
                print(self.action_dict[actions[i][0]], self.id, actions[i][1])
            else:
                print(self.action_dict[actions[i][0]], self.id)

    # 前往目的工作台  对于导航算法也有提升空空间
    def go_to_workbench(self, tar_id, tar_workbench, start = False):
        tar_x, tar_y = float(tar_workbench[0]), float(tar_workbench[1])
        # 计算偏航角
        yaw = self.clc_yaw(tar_x - self.x, tar_y - self.y)
        # 如果偏航角 > 3/200 * PI 减速 最大速转向
        if yaw > 3 * np.pi / 200:
            # 调整角度
            self.adopt_action([[1, 4]])
            # 减速
            self.adopt_action([[0, 9999]])
        # 如果偏航角 < 3/200 * PI 加速 微调转向
        else:
            # 调整角度
            self.adopt_action([[1, yaw / 0.15]])
            # 前进四！
            self.adopt_action([[0, 9999]])

        if int(self.pos_workbench_id) == int(tar_id):  # 到达目的地:
            # sys.stderr.write(str((self.pos_workbench_id)) + " " + str(tar_id) + "\n")
            if start:
                self.task[1] = -1
                self.adopt_action([[2]])
            else:
                self.task[0] = -1
                self.task[2] = -1
                self.adopt_action([[3]])
        return not start and int(self.pos_workbench_id) == int(tar_id)

    # 计算偏航角
    def clc_yaw(self, tar_x, tar_y):
        a = np.array([np.cos(self.orient), np.sin(self.orient)])
        b = np.array([tar_x, tar_y])
        # 夹角cos值
        cos_ = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # 夹角sin值
        sin_ = np.cross(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        yaw = np.arctan2(sin_, cos_)
        return yaw

    # 状态更新
    def set_state(self, pos_workbench_id, orient, x, y):  # 设置xy的坐标
        self.pos_workbench_id = int(pos_workbench_id) + 1
        self.orient = float(orient)
        self.x = float(x)
        self.y = float(y)
        return

    # 分配任务
    def set_task(self, task_type, start, end):
        self.task = [task_type, start, end]
        return


if __name__ == '__main__':
    # 初始化工作台  每个工作台的位置都是独一无二的
    workbench = {i: [] for i in range(1, 10)}  # 工作台坐标
    cnt_workbench = [0 for i in range(10)]  # 同种类工作台数量记录

    # 初始化智能体
    agent = [Agent(id) for id in range(4)]
    # 获取地图
    map = read_util_ok(initial=True)
    bench_id = 0
    for row in range(100):
        for col in range(100):
            if map[row][col] != '.' and map[row][col] != 'A':
                workbench[int(map[row][col])].append([bench_id, 0.25 + 0.5 * col, 0.25 + 0.5 * (99 - row)])
                # sys.stderr.write(str(bench_id) + " " + str(0.25 + 0.5 * col) + ' ' + str(0.25 + 0.5 * (99 - row)) + '\n')
                bench_id += 1
    for k, v in workbench.items():
        cnt_workbench[k] = len(v)
    num_workbench = sum(cnt_workbench)  # 工作台总数
    print('OK')  # 输出OK 表示初始化结束
    sys.stdout.flush()  # 别忘了flush一下标准输出
    try:
        while True:
            # 首先输出读入的第一个整数: 帧ID
            frame_id = input().split(' ')[0]
            print(frame_id)

            # 忽略其他输入数据，读到OK为止
            state = read_util_ok()
            # for i in range(len(state)):
            #     sys.stderr.write(" ".join(state[i]) + "\n")
            # 更新 agent的坐标信息
            for i in range(4):
                agent[i].set_state(state[i - 4][0], state[i - 4][7], state[i - 4][8], state[i - 4][9])

            # 任务栏更新
            status_dict ={
                4: [1, 2],
                5: [1, 3],
                6: [2, 3],
                7: [4, 5, 6],
                8: [7],
                9: [i for i in range(1, 8)]
            }
            start_set = {i: [] for i in range(1, 9)}  # 可选起点
            end_set = {i: [] for i in range(1, 9)}  # 可选终点
            for i in range(1, 1 + num_workbench):
                wb_message = state[i]
                type_bench = int(wb_message[0])
                if wb_message[-1] == '1':
                    start_set[type_bench].append(i)
                if type_bench > 3:
                    product_status = int(wb_message[-2])
                    for j in status_dict[type_bench]:
                        if product_status & (1 << j) == 0:
                            end_set[j].append(i)

            # sys.stderr.write("起点栏展示\n")
            # for k, v in start_set.items():
            #     sys.stderr.write(str(k) + " ".join(str(v)) + "\n")

            # 任务确定
            for id in range(4):
                if sum(agent[id].task) == -3:  # 智能体空闲，进行任务分配
                    # 分配任务 前往指定地点
                    # 起点选择  倒序 优先选取贵重的物资 重点改进的地方
                    task_type, start, end = -1, -1, -1
                    for i in range(7, 0, -1):  # state[start_set[i][-1]][0]  #
                        while end_set[i] != [] and end_is_valid(i, end_set[i][-1], id, agent):  # 满足某个条件
                            end_set[i].pop()
                        if start_set[i] != [] and end_set[i] != []:
                            task_type = i
                            start = start_set[i].pop()
                            end = end_set[i].pop()
                            break
                    agent[id].set_task(task_type, start, end)
                if agent[id].task[1] != -1:
                    agent[id].go_to_workbench(agent[id].task[1], state[agent[id].task[1]][1:3], start=True)
                elif agent[id].task[2] != -1:
                    agent[id].go_to_workbench(agent[id].task[2], state[agent[id].task[2]][1:3])
            # sys.stderr.write(" ".join(str(agent[0].task)) + "\n")

            print('OK')  # 当前帧控制指令结束，输出OK
            sys.stdout.flush()  # flush标准输出，以便判题器及时接收数据
    except EOFError:  # 读到EOF时, 程序结束
        pass

