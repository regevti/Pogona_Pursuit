from strikes import TrialStrikes
from pose import PoseAnalyzer
from fpdf import FPDF
import pandas as pd
import numpy as np
import pickle
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

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
            # data = TrialStrikes(ld).strikes_summary(is_plot=False, use_cache=True)
            # for d in data:
            #     [d.pop(f, None) for f in fields2drop]
            #     d.update({k: v for k, v in ld.info.items() if not k.startswith('block')})
            #     d['loader_id'] = loader_id
            #     l.append(d)
            l.append({k: v for k, v in ld.info.items() if not k.startswith('block')})

        info_df = pd.DataFrame(l)
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
    def group_plot(plot_func, glds, ax, xlim, ylim, is_invert_y, cmaps=None):
        for i, ld in enumerate(glds):
            cmap = cmaps[i] if cmaps else None
            plot_func(ld, ax, cmap=cmap)
        ax.set_xlim(list(xlim))
        ax.set_ylim(list(ylim))
        if is_invert_y:
            ax.invert_yaxis()

    def subplot(self, plot_func, xlim=(0, 2300), ylim=(0, 900), is_invert_y=True, is_time_cmap=False):
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
                cmaps = self.time_cmap(glds, fig, axes[ia]) if is_time_cmap else None
                self.group_plot(plot_func, glds, axes[ia], xlim, ylim, is_invert_y, cmaps=cmaps)
                if main_group != groupby[0]:
                    axes[ia].set_title(', '.join([f'{g}={v}' for g, v in zip(groupby, list(group_values))]))

    @staticmethod
    def time_cmap(glds, fig, ax):
        colormaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        z = np.linspace(0, len(colormaps) - 1, len(glds))
        cmaps = [colormaps[int(np.floor(j))] for j in z]
        custom_cmap = ListedColormap([plt.get_cmap(name)(0.5) for name in colormaps])
        time_lds = [glds[cmaps.index(c)] for c in colormaps if c in cmaps]
        cbar = fig.colorbar(ScalarMappable(norm=None, cmap=custom_cmap), ax=ax, ticks=np.linspace(0,1,len(time_lds)))

        cbar.ax.set_yticklabels([tld.day for tld in time_lds])

        return cmaps

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
        def _plot_projected_strikes(ld, ax, *args):
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
        def _plot_pd(ld, ax, *args):
            ts = TrialStrikes(ld)
            for s in ts.strikes:
                if not s.pd or not s.bug_speed:
                    continue
                ax.scatter(s.bug_speed, s.pd, color='b')
                ax.set_xlabel('bug speed [cm/sec]')
                ax.set_ylabel('PD [cm]')

        self.subplot(_plot_pd, xlim=xlim, ylim=ylim, is_invert_y=False)

    def plot_arena_trajectory(self, xlim=(0, 1400), ylim=(0, 1100), **kwargs):
        def _plot_arena_trajectory(ld, ax, cmap):
            a = PoseAnalyzer(ld)
            a.arena_trajectories(ax=ax, cmap=cmap, **kwargs)

            rect = patches.Rectangle((70, 940), 1270, 50, linewidth=1, edgecolor='k', facecolor='k')
            ax.add_patch(rect)

        self.subplot(_plot_arena_trajectory, xlim=xlim, ylim=ylim, is_invert_y=False, is_time_cmap=True)