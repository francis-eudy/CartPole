import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# Set Up OpenAI Gym Env

# obs is a numpy consisting of 4 floats
# 1) Horizontal Position (0.0 = center)
# 2) Velocity
# 3) Angle of the Pole (0.0 = vertical)
# 4) Angular Velocity
# print(obs)
# env.render()

# Number of Possible actions in the env
# print(env.action_space)

# action = 1  # Go Right
# obs, reward, done, info = env.step(action)
# print(obs)  # New observation
# print(reward)  # reward of 1 for staying up right
# print(done)  # True when the session is over
# print(info)  # Extra Debug Info on the Env

# def basic_policy(obs):
#     angle = obs[2]
#     return 0 if angle < 0 else 1
#
# totals = []
# for episode in range(500):
#     episode_rewards = 0
#     obs = env.reset()
#     for step in range(1000):  # 1000 steps max, we don't want to run forever
#         # env.render()
#         action = basic_policy(obs)
#         obs, reward, done, info = env.step(action)
#         episode_rewards += reward
#         if done:
#             break
#     totals.append(episode_rewards)
#
# print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
env = gym.make("CartPole-v0")
obs = env.reset()

# Define the Neural Network Architecture
n_inputs = env.observation_space.shape[0]  # 4 in this case
n_hidden = n_inputs  # rather simple network do not need more hidden layers
n_outputs = 1  # Probability of accelerating left

# Old
#initializer = tf.contrib.layers.variance_scaling_initializer()
# New
# initializer = tf.keras.initializers.VarianceScaling()



# Set the Learning Rate
learning_rate = 0.01

# Build the Neural Network
# Old
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer='VarianceScaling')
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer='VarianceScaling')
outputs = tf.nn.sigmoid(logits)
# New
# input_layer = tf.keras.layers.InputLayer(input_shape=(n_inputs,))
# hidden_layer = tf.keras.layers.Dense(input_layer, n_hidden, activation=tf.nn.elu, kernel_initializer="VarianceScaling")
# logits_layer = tf.keras.layers.Dense(hidden_layer, n_outputs, kernel_initializer="VarianceScaling")
# output_layer = tf.keras.layers.Dense(1, activation='sigmoid')


# Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, discount_rate):
    discount_rewards = np.empty(len(rewards))
    cumlative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumlative_rewards = rewards[step] + cumlative_rewards * discount_rate
        discount_rewards[step] = cumlative_rewards
    return discount_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discount_rewards - reward_mean) / reward_std for discount_rewards in all_discounted_rewards]


# Training Policy
n_iterations = 150  # number of training iterations
n_max_steps = 200  # max steps per episode   env limits to 200 now
n_games_per_update = 10  # train the policy every 10 episodes
save_iterations = 10  # save the model every 10 training iterations
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []  # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        for game in range(n_games_per_update):
            current_rewards = []  # all raw rewards from the current episode
            current_gradients = []  # all gradients from the current episode
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run(
                    [action, gradients],
                    feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # Update the Policy
        print("Iteration:", iteration)
        totals = [np.sum(t) for t in all_rewards]
        print(">", len(totals))
        print(">", totals)
        print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")




env.env.close()




if __name__ == "__main__":
    pass
