import numpy as np

class FIMTDD(object):
    def __init__(self, **args):
        self.left = LeafNode(parent = self, can_grow=True, **args)

    def eval(self,x):
        return self.left.eval(np.array(x).reshape((-1,)))

    def eval_and_learn(self,x,y):
        return self.left.eval_and_learn(np.array(x).reshape((-1,)), np.array(y).reshape((-1,)))

    def __repr__(self):
        return "FIMTDD:\n" + self.left._to_str(0)


class Node(object):
    def __init__(self,data_dim=1, parent=None, left=None, right=None, alpha=0.005, threshold=50, n_min=100, gamma=0.01, learn=0.1, decay_rate=0.995, isAlt=False):
        """
        :param parent:      The parent of the node
        :param left:        left child-node
        :param right:       right child-node
        :param alpha:       alpha value for change-detection
        :param threshold:   threshold for change detection
        :param n_min:       minimum time period for split and subtree replacement
        :param gamma:       value for the hoefding-bound
        :param learn:       learning rate of the leafnote
        :param decay_rate:  decay rate of the exponential running average of squared errors
        :return:
        """
        self.data_dim = data_dim
        self.parent = parent
        self.alpha = alpha
        self.threshold = threshold
        self.n_min = n_min
        self.gamma = gamma
        self.learn = learn
        self.decay_rate = decay_rate
        self.isAlt = isAlt
        self.q = 0

        self.m = 0.0
        self.C = 0.0

        # Number of datapoints used for learning
        self.samples_seen = 0

        # Alternative tree starting at this node (if it exists)
        self.alt_tree = None

    # eval and eval_and_learn methods must be defined for a subclass of Node
    def eval(self,x):
        raise NotImplementedError("This function must be implemented by a subclass!")
    def eval_and_learn(self,x,y):
        raise NotImplementedError("This function must be implemented by a subclass!")

    def grow_alt_tree(self, can_grow):
        self.alt_tree = SplitNode(key=(self.m+np.random.randn()*self.C)/self.samples_seen, key_dim=0, parent=self.parent, isAlt=True, can_grow=can_grow, **self.node_kwargs)


    def check_swap_alt_tree(self):
        if self.alt_tree and self.alt_tree.samples_seen%self.n_min == 0 and self.alt_tree.samples_seen != 0:
            #check all n_min samples the q statistics of current and alt-tree
            if self.q > self.alt_tree.q:
                #if alt-tree has better performance, replace this node with alternate subtree
                if self == self.parent.left:
                    self.parent.left = self.alt_tree
                elif self == self.parent.right:
                    self.parent.right = self.alt_tree

                # The alt tree is now no longer an alt tree
                self.alt_tree.isAlt = False
                # The direct next level of leaves could now grow
                if self.alt_tree.left and isinstance(self.alt_tree.left, LeafNode):
                    self.alt_tree.left.can_grow = True
                if self.alt_tree.right and isinstance(self.alt_tree.right, LeafNode):
                    self.alt_tree.right.can_grow = True

            if self.alt_tree.samples_seen >= self.n_min*10:
                #if alternate tree is still not better than the current one, remove it
                self.alt_tree = None


class SplitNode(Node):
    def __init__(self, key, key_dim, parent=None, isAlt=False, can_grow=False, **kwargs):
        """
        :param key:         the key of the node
        :param key_dim:     indicates the attribute of the key value
        :param parent:      The parent of the node
        :param alpha:       alpha value for change-detection
        :param threshold:   threshold for change detection
        :param n_min:       minimum time period for split and subtree replacement
        :param gamma:       value for the hoefding-bound
        :param learn:       learning rate of the leafnote
        :return:
        """
        self.node_kwargs = kwargs.copy()

        self.key = key
        self.key_dim = key_dim
        self.left = LeafNode(parent=self, can_grow=can_grow, **self.node_kwargs)
        self.right = LeafNode(parent=self, can_grow=can_grow, **self.node_kwargs)

        # the remaining arguments are just standard Node properties
        super().__init__(parent=parent, isAlt=isAlt, **self.node_kwargs)

    def eval(self,x):
        """
        :param x:   data point
        :return:    prediction
        """
        return self.left.eval(x) if x[self.key_dim] <= self.key else self.right.eval(x)


    def eval_and_learn(self,x,y):
        """
        :param x:   data point
        :param y:   label
        :return:    prediction
        """
        # increment seen sample counter
        self.samples_seen += 1

        self.m += x
        self.C += np.outer(x,x)

        # pass data on to appropriate child and get prediction
        yp = self.left.eval_and_learn(x,y) if x[self.key_dim] <= self.key else self.right.eval_and_learn(x,y)

        # pass data on to the alt tree, too, should it exist
        if self.alt_tree:
            self.alt_tree.eval_and_learn(x,y)

        # update squared error
        sq_loss = (y - yp)**2

        # update exponential running average of squared error and cumulative squared error
        self.q = sq_loss if self.samples_seen == 1 else self.decay_rate*self.q + (1-self.decay_rate)*sq_loss

        # check if this should be replaced by the alt-tree (should it exist)
        self.check_swap_alt_tree()

        # # start an alt-tree that can grow if both child nodes want to grow an alt-tree and this is not an alt-tree already
        # if self.left.alt_tree and self.right.alt_tree and not self.isAlt
        #     self.alt_tree = LeafNode(parent=self.parent, isAlt=True, can_grow=True, **self.node_kwargs)

        return yp

    def _to_str(self, indent_level):
        return self.left._to_str(indent_level+1)+"\t"*indent_level + "+ Split at x[{}]={} {} {} samples seen, q={}\n".format(self.key_dim, self.key, "(alternative)" if self.isAlt else "", self.samples_seen, self.q)+self.right._to_str(indent_level+1) + (self.alt_tree._to_str(indent_level) if self.alt_tree else "")

class LeafNode(Node):
    """
    LeafNode-Object for FIMTDD
    """
    def __init__(self, model=None, parent=None, isAlt=False, can_grow=False, **kwargs):
        """
        :param model:       the model to be used in the leaf node
        :param parent:      The parent of the node
        :param can_grow:     whether or not this leaf can grow (split)
        :param alpha:       alpha value for change-detection
        :param threshold:   threshold for change detection
        :param n_min:       minimum time period for split and subtree replacement
        :param gamma:       value for the hoefding-bound
        :param learn:       learning rate of the leafnote
        :return:
        """
        self.node_kwargs = kwargs.copy()

        self.can_grow = can_grow
        self.alt_tree = None

        # the remaining arguments are just standard Node properties
        super().__init__(parent=parent, isAlt=isAlt, **self.node_kwargs)

        self.model = model if model else LinearRegressor(self.data_dim)


    def eval(self,x):
        """
        :param x:   data point
        :return:    prediction
        """
        return self.model.eval(x)


    def eval_and_learn(self,x,y):
        """
        :param x:   data point
        :param y:   label
        :return:    prediction
        """
        # increment seen sample counter
        self.samples_seen += 1

        self.m += x
        self.C += np.outer(x,x)

        # update model and get prediction
        yp = self.model.eval_and_learn(x, y)

        # pass data on to the alt tree, too, should it exist
        if self.alt_tree:
            self.alt_tree.eval_and_learn(x,y)

        # update squared error
        sq_loss = (y - yp)**2

        # update exponential running average of squared error and cumulative squared error
        self.q = sq_loss if self.samples_seen == 1 else self.decay_rate*self.q + (1-self.decay_rate)*sq_loss

        # check if this should be replaced by the alt-tree (should it exist)
        self.check_swap_alt_tree()

        # start a non-growing alt-tree if possible, there isn't one already and we've seen enough
        if self.can_grow and not self.alt_tree and self.samples_seen >= self.n_min:
            self.grow_alt_tree(can_grow=False)

        return yp

    def _to_str(self, indent_level):
        return "\t"*indent_level + "o Leaf ({} samples seen, q={})\n".format(self.samples_seen, self.q) + (self.alt_tree._to_str(indent_level) if self.alt_tree else "")

class LinearRegressor(object):
    def __init__(self, dim):
        self.w = np.zeros(dim+1)
        self.C = np.eye(dim)
        self.samples_seen = 0

    def eval(self, x):
        return np.hstack([x, 1]).dot(self.w)

    def eval_and_learn(self, x, y):
        yp = self.eval(x)
        self.w[-1] = (self.w[-1]*self.samples_seen + y)/(self.samples_seen+1)
        self.samples_seen += 1
        return yp
