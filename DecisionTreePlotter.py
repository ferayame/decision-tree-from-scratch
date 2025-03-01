import matplotlib.pyplot as plt

class DtPlotter:
    def __init__(self, tree):
        self.tree = tree

    def _get_tree_depth(self, tree):
        """Calculate the depth of the tree."""
        if tree["leaf"]:
            return 0
        return max(self._get_tree_depth(tree["left"]), self._get_tree_depth(tree["right"])) + 1

    def _plot_node(self, ax, node_text, center_pos, parent_pos, node_type):
        """Draw a node with the given text and position."""
        ax.annotate(
            node_text,
            xy=parent_pos,
            xycoords='axes fraction',
            xytext=center_pos,
            textcoords='axes fraction',
            va="center",
            ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="black"),
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"),
        )

    def _plot_tree(self, ax, tree, parent_pos, node_pos, depth, x_offset, depth_offset):
        """Recursively plot the tree."""
        if tree["leaf"]:
            node_text = f"Class: {tree['value']}"
            self._plot_node(ax, node_text, node_pos, parent_pos, "leaf")
            return

        # Plot decision node
        node_text = f"X[{tree['feature_index']}] <= {tree['threshold']:.2f}"
        self._plot_node(ax, node_text, node_pos, parent_pos, "decision")

        # Plot left and right subtrees
        self._plot_tree(
            ax,
            tree["left"],
            node_pos,
            (node_pos[0] - x_offset / 2**depth, node_pos[1] - depth_offset),
            depth + 1,
            x_offset,
            depth_offset,
        )
        self._plot_tree(
            ax,
            tree["right"],
            node_pos,
            (node_pos[0] + x_offset / 2**depth, node_pos[1] - depth_offset),
            depth + 1,
            x_offset,
            depth_offset,
        )

    def plot(self):
        """Plot the decision tree."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # Calculate tree depth and initial positions
        tree_depth = self._get_tree_depth(self.tree)
        x_offset = 1.0
        depth_offset = 1.0 / (tree_depth + 1)

        # Start plotting from the root
        self._plot_tree(ax, self.tree, (0.5, 1.0), (0.5, 1.0), 1, x_offset, depth_offset)

        plt.show()