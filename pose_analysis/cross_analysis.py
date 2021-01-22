from strikes import TrialStrikes
from pose import PoseAnalyzer
from fpdf import FPDF
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

STRIKE_FIELDS = ['bug_type']


class MultiStrikesAnalyzer:
    def __init__(self, loaders, groupby=None, main=None, **filters):
        if groupby:
            assert isinstance(groupby, dict), 'groupby must be dictionary'
            assert all(isinstance(v, list) for v in groupby.values() if v is not None), \
                'all groupby values must be list or None'
        if main:
            assert main in groupby, f'main {main} is not in groupby'
        self.loaders = self.filter(loaders, **filters)
        self.main = main
        self.groupby = groupby
        self.info_df = self.load_data()

    def load_data(self):
        l = []
        fields2drop = ['bug_traj', 'nose']
        for ld in self.loaders:
            data = TrialStrikes(ld).strikes_summary(is_plot=False, use_cache=True)
            for d in data:
                [d.pop(f, None) for f in fields2drop]
                d.update({k: v for k, v in ld.info.items() if not k.startswith('block')})
                l.append(d)

        info_df = pd.DataFrame(l)
        info_df.is_anticlockwise.fillna(False, inplace=True)
        return info_df

    @staticmethod
    def filter(loaders, **filters):
        if not filters:
            return loaders
        lds = []
        for ld in loaders:
            if all(ld.info.get(filter) == value for filter, value in filters.items()):
                lds.append(ld)
        print(f'Left with {len(lds)} loaders after applying the filters')
        return lds

    @staticmethod
    def create_subplots(n_groups):
        cols = min([4, n_groups or 1])
        rows = int(np.ceil((n_groups or 1) / cols))
        fig = plt.figure(figsize=(20, rows * 5))
        axes = fig.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        return fig, axes

    @staticmethod
    def group_plot(plot_func, glds, ax, xlim, ylim, is_invert_y):
        for ld in glds:
            plot_func(ld, ax)
        ax.set_xlim(list(xlim))
        ax.set_ylim(list(ylim))
        if is_invert_y:
            ax.invert_yaxis()

    def subplot(self, plot_func, xlim=(0, 2300), ylim=(0, 900), is_invert_y=True):
        if not self.groupby:
            fig, axes = self.create_subplots(1)
            self.group_plot(plot_func, self.loaders, axes[0], xlim, ylim, is_invert_y)
            return

        groupby = list(self.groupby.keys())
        main_group = self.main or groupby[0]
        if len(groupby) > 1:
            groupby.remove(main_group)
        main_values = self.groupby[main_group] if self.groupby.get(main_group) else self.info_df[main_group].unique()
        for main_group_value in main_values:
            groups = self.info_df[self.info_df[main_group] == main_group_value].groupby(groupby).groups
            groups = self.check_groups(groups, groupby)
            fig, axes = self.create_subplots(len(groups))
            fig.suptitle(f'{main_group} = {main_group_value}', fontsize=15)
            for ia, (group_values, group_idx) in enumerate(groups.items()):
                if len(groupby) == 1:
                    group_values = [group_values]
                glds = [ld for j, ld in enumerate(self.loaders) if j in group_idx]
                self.group_plot(plot_func, glds, axes[ia], xlim, ylim, is_invert_y)
                if main_group != groupby[0]:
                    axes[ia].set_title(', '.join([f'{g}={v}' for g, v in zip(groupby, list(group_values))]))

    def check_groups(self, groups, groupby):
        new_groups = {}
        for group_values, group_idx in groups.items():
            is_group_ok = True
            for key, value in zip(groupby, group_values if len(groupby) > 1 else [group_values]):
                if self.groupby.get(key) and value not in self.groupby[key]:
                    is_group_ok = False
            if is_group_ok:
                new_groups[group_values] = group_idx

        return new_groups

    def create_strikes_pdf(self, filename: str):
        assert filename.endswith('.pdf'), 'filename must end with .pdf'
        image_list = []
        for ld in self.loaders:
            ts = TrialStrikes(ld)
            data = ts.strikes_summary(is_plot=False, use_cache=True)
            image_list += [d['save_image_path'] for d in data if d.get('save_image_path')]

        pdf = FPDF()
        for i in np.arange(0, len(image_list), 2):
            pdf.add_page()
            w = 200
            h = 145
            pdf.image(image_list[i], 5, 2, w, h)
            if i < len(image_list) - 1:
                pdf.image(image_list[i + 1], 5, 2 + h, w, h)

        pdf.output(filename, "F")

    def plot_projected_strikes(self, xlim=(-200, 200), ylim=(-200, 200)):
        def _plot_projected_strikes(ld, ax):
            ts = TrialStrikes(ld)
            for s in ts.strikes:
                if 'bug_type' in self.groupby and not ax.patches:
                    ax.add_patch(plt.Circle((0, 0), s.bug_radius, color='lemonchiffon', alpha=0.4))
                if s.bug_traj is None:
                    continue
                s.plot_projected_strike(ax, is_plot_strike_only=True)
                ax.plot([0, 0], ylim, 'k')
                ax.plot(xlim, [0, 0], 'k')

        self.subplot(_plot_projected_strikes, xlim=xlim, ylim=ylim, is_invert_y=False)

    def plot_pd(self, xlim=(0, 15), ylim=(-2.5, 5)):
        def _plot_pd(ld, ax):
            ts = TrialStrikes(ld)
            for s in ts.strikes:
                if not s.pd or not s.bug_speed:
                    continue
                ax.scatter(s.bug_speed, s.pd, color='b')
                ax.set_xlabel('bug speed [cm/sec]')
                ax.set_ylabel('PD [cm]')

        self.subplot(_plot_pd, xlim=xlim, ylim=ylim, is_invert_y=False)

    def plot_arena_trajectory(self, xlim=(0, 1000), ylim=(0, 1000)):
        def _plot_arena_trajectory(ld, ax):
            a = PoseAnalyzer(ld.video_path)
            a.arena_trajectories(ax=ax, cmap='Oranges')

        self.subplot(_plot_arena_trajectory, xlim=xlim, ylim=ylim, is_invert_y=False)