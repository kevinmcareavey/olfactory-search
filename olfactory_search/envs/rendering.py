import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pygame


# def get_scalarmap(index, Nhits=0):
#     if index == 0:
#         topcolors = colormaps.get_cmap('Greys')
#         bottomcolors = colormaps.get_cmap('Spectral_r')
#         newcolors = np.vstack((topcolors(0.5),
#             bottomcolors(np.linspace(0, 1, Nhits - 1))))
#         cmap = ListedColormap(newcolors, name='GreyColors')
#         cmap.set_under(color="black")
#         sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=Nhits - 0.5), cmap=cmap)
#     elif index == 1:
#         cmap = colormaps.get_cmap("viridis")
#         sm = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap)
#     return cmap, sm

# @dataclasses.dataclass
# class VisitationMap:
#     grid_size: list[int, int] | list[int, int, int]

#     """
#     Returns: a np.array of shape (grid_size, grid_size) with the number of visits to each cell
#     """
#     def _reset(self):
#         num_states = np.prod(self.grid_size)
#         self.data = np.zeros((num_states, 1), dtype=np.int64)

#     def _add_visit(self, state):
#         assert len(state) == len(self.grid_size)

#         strides = [1]
#         for dim in reversed(self.grid_size[:-1]):
#             strides.insert(0, strides[0] * dim)
#         index = sum(i * s for i, s in zip(state, strides))
#         self.data[index] += 1

#     @property
#     def _total_visit(self):
#         return np.sum(self.data)

# class RenderFrame(object):
#     """
#     There are three axis for visualization: 
#         - ax1 shows the trajectory and the true plume model; 
#         - ax2 shows the agent's belief (optional) of the plume model.
#     """
#     def __init__(self, source, agent, isDrawSource=True):
#         self.source = source
#         self.agent = agent
#         self.isDrawSource = isDrawSource
#         self.Nhits = 3

#         self.ax = None
#         self.history_states = []
#         self.history_hits = []

#     def setup(self):
#         figsize = (10, 10)
#         Nhits = self.Nhits
#         cm = []

#         # setup figure
#         fig, ax = plt.subplots(1, 2, figsize=figsize)
#         bottom = 0.1
#         top = 0.88
#         left = 0.05
#         right = 0.94
#         plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.35)

#         # axis for state and p_source
#         for i in range(2):
#             ColorMap, ScalarMap = get_scalarmap(i, Nhits)
#             cm.append(ColorMap)

#             divider = make_axes_locatable(ax[i])
#             cax = divider.append_axes("right", size="5%", pad=0.3)
#             fig.colorbar(ScalarMap, cax=cax, ticks=np.arange(0, Nhits))
#             ax[i].set_aspect("equal", adjustable="box")
#             # ax[i].axis("off")

#         # position of source
#         if self.isDrawSource:
#             for i in range(2):
#                 ax[i].plot(self.source[0], self.source[1], color="r", marker="$+$", markersize=8, zorder=10000)

#         # trajectory
#         for i in range(2):
#             ax[i].plot(self.agent[0], self.agent[1], color="r", marker='o', markersize=8, zorder=10000)

#         self.ax = ax
#         self.cm = cm
#         self.history_states.append(self.agent)
#         self.history_hits.append(0)

#     def render_current_step(
#             self,
#             new_pos: np.ndarray,
#             new_hit: int,
#     ):
#         self.grid_size = 19
#         for i in range(2):

#             self.ax[i].scatter(new_pos[0], new_pos[1], 70, c=new_hit, cmap=self.cm[0], marker="o", alpha=0.5)
#             self.ax[i].plot([self.history_states[-1][0], new_pos[0]], [self.history_states[-1][1], new_pos[1]], c="k", linewidth=1)

#         self.history_states.append(new_pos)
        


class Visualization:
    """ A class for visualizing the search in 1D, 2D or 3D

    Args:
        env (SourceTracking):
            an instance of the SourceTracking class
        live (bool, optional):
            whether to show live preview (faster if False) (default=False)
        filename (str, optional):
            file name for the video (default='test')
        log_prob (bool, optional):
            whether to show log(prob) instead of prob (default=False)
        marginal_prob_3d (bool, optional):
            in 3D, whether to show marginal pdfs on each plane, instead of the pdf in the planes that the
            agent crosses (default=False)
    """
    
    def __init__(self,
                 Ndim: int,
                 source,
                 agent,
                 log_prob=True,
                 marginal_prob_3d=False,
                 ):
        if Ndim > 3 or Ndim < 1 or not isinstance(Ndim, int):
            raise Exception("Problem with Ndim: visualization is not possible")
        
        self.Ndim = Ndim
        self.source = source
        self.agent = agent

        self.is_draw_source = True

        self.log_prob = log_prob
        self.marginal_prob_3d = marginal_prob_3d

    def record_snapshot(self, num, toptext=''):
        """Create a frame from current state of the search, and save it.

        Args:
            num (int): frame number (used to create filename)
            toptext (str): text that will appear in the top part of the frame (like a title)
        """

        if self.video_live:
            if not hasattr(self, 'fig'):
                fig, ax = self._setup_render()
                # ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
                # ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)
                self.fig = fig
                self.ax = ax
            else:
                fig = self.fig
                ax = self.ax
                # ax[0].title.set_text("observation map (current: %s)" % self._obs_to_str())
                # ax[1].title.set_text("source probability distribution (entropy = %.3f)" % self.env.entropy)
        else:
            fig, ax = self._setup_render()
            # ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
            # ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)

        self._update_render(fig, ax, toptext=toptext)

        if self.video_live:
            plt.pause(0.1)
        plt.draw()
        framefilename = self._framefilename(num)
        fig.savefig(framefilename, dpi=150)
        if not self.video_live:
            plt.close(fig)

    def _setup_render(self):

        figsize = (12.5, 5.5)

        if self.Ndim == 2:
            # setup figure
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            bottom = 0.1
            top = 0.88
            left = 0.05
            right = 0.94
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.35)

            # state
            cmap0 = self._cmap0()
            sm0 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=self.Nhits - 0.5), cmap=cmap0)
            divider = make_axes_locatable(ax[0])
            cax0 = divider.append_axes("right", size="5%", pad=0.3)
            fig.colorbar(sm0, cax=cax0, ticks=np.arange(0, self.Nhits))
            ax[0].set_aspect("equal", adjustable="box")
            ax[0].axis("off")

            # p_source
            cmap1 = self._cmap1()
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            divider = make_axes_locatable(ax[1])
            cax1 = divider.append_axes("right", size="5%", pad=0.3)
            if self.log_prob:
                cbar1 = fig.colorbar(sm1, cax=cax1, extend="min")
            else:
                cbar1 = fig.colorbar(sm1, cax=cax1)
            if self.video_live:
                self.cbar1 = cbar1
            ax[1].set_aspect("equal", adjustable="box")
            ax[1].axis("off")

            # position of source
            if self.is_draw_source:
                for i in range(2):
                    ax[i].plot(self.source[0], self.source[1], color="r", marker="$+$", markersize=8, zorder=10000)

        return fig, ax

    def _update_render(self, fig, ax, toptext=''):

        if self.video_live:
            if hasattr(self, 'artists'):
                for artist in range(len(self.artists)):
                    if self.artists[artist] is not None:
                        if isinstance(self.artists[artist], list):
                            for art in self.artists[artist]:
                                art.remove()
                        else:
                            self.artists[artist].remove()

        if self.Ndim == 2:
            self._draw_2D(fig, ax)

        # bottomtext = "$\mathcal{L} = \lambda / \Delta x = $" + str(self.env.lambda_over_dx) \
        #              + "$\qquad$ $\mathcal{I} = R \Delta t = $" + str(self.env.R_dt) \
        #              + "$\qquad$ $h_{\mathrm{init}}$ = " + str(self.env.initial_hit)
        # sup = plt.figtext(0.5, 0.99, toptext, fontsize=13, ha="center", va="top")
        # bot = plt.figtext(0.5, 0.01, bottomtext, fontsize=10, ha="center", va="bottom")
        # if self.video_live:
        #     self.artists.append(sup)
        #     self.artists.append(bot)

    def _draw_2D(self, fig, ax):
        # hit map
        cmap0 = self._cmap0()
        img0 = ax[0].imshow(
            np.transpose(self.env.hit_map),
            vmin=-0.5,
            vmax=self.env.Nhits - 0.5,
            origin="lower",
            cmap=cmap0,
        )

        # p_source
        cmap1 = self._cmap1()
        img1 = ax[1].imshow(
            np.transpose(self.env.p_source),
            vmin=np.min(self.env.p_source),
            vmax=np.max(self.env.p_source),
            origin="lower",
            aspect='equal',
            cmap=cmap1,
        )
        if self.video_live:
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            self.cbar1.update_normal(sm1)

        # position of agent
        aloc = [0] * 2
        for i in range(2):
            aloc[i] = ax[i].plot(self.agent[0], self.agent[1], "ro")

        if self.video_live:
            self.artists = [img0, img1] + [a for a in aloc]

    def _cmap0(self):
        topcolors = plt.cm.get_cmap('Greys', 128)
        bottomcolors = plt.cm.get_cmap('Spectral_r', 128)
        newcolors = np.vstack((topcolors(0.5),
                               bottomcolors(np.linspace(0, 1, self.Nhits - 1))))
        cmap0 = ListedColormap(newcolors, name='GreyColors')
        if self.Ndim == 2:
            cmap0.set_under(color="black")
        return cmap0

    def _cmap1(self):
        if self.Ndim == 2:
            cmap1 = plt.cm.get_cmap("viridis", 50)
        return cmap1

    @property
    def _alpha0(self):
        return None
    
    @property
    def _alpha1(self):
        return None







if __name__ == "__main__":

    rf = RenderFrame(
        source=[5, 5],
        agent=[0, 0],
        isDrawSource=True
    )
    rf.setup()

    traj = [np.array([1, 0]), np.array([2, 0]), np.array([2, 1]),
            np.array([3, 1]), np.array([3, 2]), np.array([3, 3])]

    for i in range(6):
        rf.render_current_step(
            new_pos = traj[i],
            new_hit = 10
        )
    plt.show()

# def render_window_2d(self):
#     img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

#     if self.render_mode == "human":
#         img = np.transpose(img, axes=(1, 0, 2))
#         if self.render_size is None:
#             self.render_size = img.shape[:2]
#         if self.window is None:
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode(
#                 (self.screen_size, self.screen_size)
#             )
#             pygame.display.set_caption("minigrid")
#         if self.clock is None:
#             self.clock = pygame.time.Clock()
#         surf = pygame.surfarray.make_surface(img)

#         # Create background with mission description
#         offset = surf.get_size()[0] * 0.1
#         # offset = 32 if self.agent_pov else 64
#         bg = pygame.Surface(
#             (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
#         )
#         bg.convert()
#         bg.fill((255, 255, 255))
#         bg.blit(surf, (offset / 2, 0))

#         bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

#         font_size = 22
#         text = self.mission
#         font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
#         text_rect = font.get_rect(text, size=font_size)
#         text_rect.center = bg.get_rect().center
#         text_rect.y = bg.get_height() - font_size * 1.5
#         font.render_to(bg, text_rect, text, size=font_size)

#         self.window.blit(bg, (0, 0))
#         pygame.event.pump()
#         self.clock.tick(self.metadata["render_fps"])
#         pygame.display.flip()