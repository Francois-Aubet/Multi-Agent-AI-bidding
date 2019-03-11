from algorithms.CEAlgorithm import CEAlgorithm
from algorithms.IAlgorithm import IAlgorithm
import numpy as np
import tensorflow as tf


class PolicyGradientGaussian(object):
    def __init__(self, config, sess):
        # initialization
        self._s = sess

        # build the graph
        self._input = tf.placeholder(tf.float32,
                                     shape=[config['state_dim']])

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        hidden_size = config['hidden_size']

        hidden_layers = tf.layers.dense(self._input,
                                        units=hidden_size[0],
                                        activation=tf.nn.sigmoid,
                                        use_bias=True,
                                        name='weight_h0')

        for h in hidden_size[1:-1]:
            hidden_layers = tf.layers.dense(hidden_layers,
                                            units=h,
                                            activation=tf.nn.sigmoid,
                                            use_bias=True,
                                            name='weight')

        hidden_layers = tf.layers.dense(hidden_layers,
                                        units=1,
                                        activation=None,
                                        use_bias=True,
                                        name='weight')

        self.mu = tf.squeeze(hidden_layers)

        with tf.variable_scope('weight', reuse=True):
            self.weights = tf.get_variable('kernel')

        self.action_dist = tf.distributions.Normal(self.mu, 1.0)

        # training part of graph
        self._action = tf.placeholder(tf.float32)
        self._reward = tf.placeholder(tf.float32)

        # get log probs of actions
        log_acts = self._action
        self.act_prob = tf.log(self.action_dist.prob(log_acts))

        # surrogate loss
        self.loss = -tf.reduce_mean(self.act_prob * self._reward)

        # update + gradient clipping
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self._train = optimizer.apply_gradients(zip(gradients, variables))

    def act(self, state):
        # get one action, by sampling
        sample = self.action_dist.sample()
        return self._s.run(sample,
                           feed_dict={self._input: state})

    def train_step(self, obs, learning_rate):
        batch_feed = {self._input: obs,
                      self.learning_rate: learning_rate}
        _, loss = self._s.run([self._train, self.loss], feed_dict=batch_feed)
        return loss


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=20,  # number of hidden units
            activation=tf.nn.relu,
            name='l1'
        )

        mu = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma + 0.01)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(tf.exp(self.normal_dist.sample(1)), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(tf.log(self.a))  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)  # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, GAMMA=0.95):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


class PolicyGradient(IAlgorithm):
    alg_name = "PolicyGrad"

    def __init__(self, config, ctrModel=None, ctr_pred=None, use_pretrained=False):
        """ Constructor of the abstract class.
            Should be called when initializing the child classes.
        """

        sess = tf.Session()

        self.actor = Actor(sess, n_features=4, lr=config['LR_A'],
                           action_bound=[config['A_LBOUND'], config['A_UBOUND']])
        self.critic = Critic(sess, n_features=4, lr=config['LR_C'])

        sess.run(tf.global_variables_initializer())

        self.ctrModel = ctrModel
        self._ctr_pred_table = ctr_pred
        self.use_pretrained = use_pretrained
        self.prev_s = None
        self.prev_a = None

    def train(self, data_handler):
        """ Training the algorithm using self._train* .
        This needs to be overwritten in every child classes.

        @:return: the train error

        use tqdm to display the progress because dataset is big.
        """
        train_x, train_y, valid_x, _, test_x = data_handler.get_datasets()

        if not self.use_pretrained or self._ctr_pred_table is None:
            self.ctrModel.train(data_handler)

        ctr_train = self.ctrModel.predict(train_x)
        self._ctr_pred_table = dict(zip(train_x.bidid.values, ctr_train))

    def predict(self, test_x):
        """ Predicting for a dataset. This needs to be overwritten in every child classes.
        @:return: the bid
        """

        return np.repeat(self._base_bid, test_x.shape[0])

    def predict_single(self, impression, metric):
        """ Predicting for a single impression. T
        """

        ctr = self._ctr_pred_table[impression.bidid.values[0]]
        n_imp_left = metric.n_impression_left
        budget = metric.budget
        market_price = impression.payprice

        s = np.array([ctr, n_imp_left, budget, market_price])

        if self.prev_a is not None:
            # learn from prev action
            r = metric.click_on_last_bid
            td_error = self.critic.learn(self.prev_s, r, s)  # gradient = grad[r + gamma * V(s_) - V(s)]
            self.actor.learn(self.prev_s, self.prev_a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        a = self.actor.choose_action(s).ravel()[0] * 80

        self.prev_s = s
        self.prev_a = a

        return a
