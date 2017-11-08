class Node(object):
    def __init__(self,parent=None, left=None, right=None, alpha=0.005, threshold=50, n_min=100, gamma=0.01, learn=0.1, decay_rate=0.995):
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
        self.parent = parent
        self.left = left
        self.right = right
        self.alpha = alpha
        self.threshold = threshold
        self.n_min = n_min
        self.gamma = gamma
        self.learn = learn
        self.decay_rate = decay_rate

        # Number of datapoints used for learning
        self.samples_seen = 0

        # Alternative tree starting at this node (if it exists)
        self.alt_tree = None

        # eval and eval_and_learn methods must be defined for a subclass of Node
        def eval(self,x):
            raise NotImplementedError("This function must be implemented by a subclass!")
        def eval_and_learn(self,x,y):
            raise NotImplementedError("This function must be implemented by a subclass!")


class SplitNode(Node):
    def __init__(self, key, key_dim, **kwargs):
        """
        :param parent:      The parent of the node
        :param key:         the key of the node
        :param key_dim:     indicates the attribute of the key value
        :param left:        left child-node
        :param right:       right child-node
        :param alpha:       alpha value for change-detection
        :param threshold:   threshold for change detection
        :param n_min:       minimum time period for split and subtree replacement
        :param gamma:       value for the hoefding-bound
        :param learn:       learning rate of the leafnote
        :return:
        """
        self.key = key
        self.key_dim = key_dim

        # children should be initialized as LeafNodes with identical parameters, but this SplitNode is their parent
        child_kwargs = kwargs.copy()
        child_kwargs["parent"] = self
        # the remaining arguments are just standard Node properties
        super().__init__(left=LeafNode(**child_kwargs), right=LeafNode(**child_kwargs), **kwargs)

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

        # pass data on to appropriate child and get prediction
        yp = self.left.eval_and_learn(x,y) if x[self.key_dim] <= self.key else self.right.eval_and_learn(x,y)

        # pass data on to the alt tree, too, should it exist
        if self.alt_tree != None:
            self.alt_tree.eval_and_learn(x,y)

        #update squared error and
        sq_loss = (y - yp)**2

        # update exponential running average of squared error and cumulative squared error
        self.q = self.decay_rate*self.q + sq_loss
        self.cum_sq_loss += sq_loss

        # check if this should be replaced by the alt-tree (should it exist)
        if self.alt_tree != None and self.alt_tree.samples_seen%self.n_min == 0 and self.alt_tree.samples_seen != 0:
            #check all n_min samples the q statistics of current and alt-tree
            if self.q > self.alt_tree.q:
                #if alt-tree has better performance, replace this node with alternate subtree
                if self == self.parent.left:
                    self.parent.left = self.alt_tree
                if self == self.parent.right:
                    self.parent.right = self.alt_tree
            if self.alt_tree.samples_seen >= self.n_min*10:
                #if alternate tree is still not better than the current one, remove it
                self.alt_tree = None

        #################################TODO: CONTINUE HERE ####################################
        # check if an alt-tree should be started
        if self.detect_change(y,yp) and self.can_grow and not self.isAlt and self.alt_tree is None:
            #avoid change detection on higher levels and grow subtree
            self.parent.can_grow = False
            self.grow_alt_tree()
        #elif not self.detection and not self.isAlt:
        #    #activate change detection if this node is not root of a subtree
        #    self.parent.detection = False
        #    self.detection = True
        if self.alt_tree != None or (not self.detection and not self.isAlt):
            #deactivate change detection on all higher level nodes if low level change detection is allready triggered
            self.parent.detection = False
            self.detection = False
        else:
            self.detection = True
        return yp

    def detect_change(self,y,yp):
        """
        Page-Hinckley-Test for change detection

        :param y:   the true label value
        :param yp:  the prediction
        :return:    true if change is detected, else false
        """
        #return False

        error = np.fabs(y-yp)
        self.cumloss += error

        self.PH += error - (self.cumloss/self.c_x) - self.alpha

        if self.minPH is None or self.PH < self.minPH :
            self.minPH = self.PH
        return self.PH - self.minPH > self.threshold
