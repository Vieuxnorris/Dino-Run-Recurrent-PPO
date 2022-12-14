<div id="header" align="center">
  <img src="https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif" width="300"/>
</div>
<h1 align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com/?lines=Hello,+There!+👋;This+is+Vieuxnorris....;Nice+to+meet+you!&center=true&size=30">
  </a>
</h1>

<hr>
<h2 align="center">🔥 Languages & Frameworks & Tools & Abilities 🔥</h2>
<br>
<p align="center">
  <code><img title="C" height="25" src="https://user-images.githubusercontent.com/26462639/200927520-387c7a12-39b1-4a98-b7ab-ad715c0f22f4.png"></code>
  <code><img title="C++" height="25" src="https://user-images.githubusercontent.com/26462639/200927221-0b418c66-7b5f-446f-add6-7951c86b9f3d.png"></code>
  <code><img title="C#" height="25" src="https://user-images.githubusercontent.com/26462639/200927585-9a825d69-afb4-4b02-b915-a5752ef026b9.png"></code>
  <code><img title="Python" height="25" src="https://user-images.githubusercontent.com/26462639/200927686-f4a77dad-d185-4cd8-8b67-6f30e5b94160.png"></code>
  <code><img title="Javascript" height="25" src="https://user-images.githubusercontent.com/26462639/200927750-c6519d36-8966-4d12-8bf5-467aed106f06.png"></code>
  <code><img title="Problem Solving" height="25" src="https://user-images.githubusercontent.com/26462639/200927922-05ab6d76-a138-4dd9-bf93-56cecee38065.png"></code>
  <code><img title="HTML5" height="25" src="https://user-images.githubusercontent.com/26462639/200927986-4827f8de-cc30-4341-8e28-33eb08c0bcf2.png"></code>
  <code><img title="CSS" height="25" src="https://user-images.githubusercontent.com/26462639/200928083-45803707-ef12-4619-acbd-56bf6df5fadb.png"></code>
  <code><img title="Git" height="25" src="https://user-images.githubusercontent.com/26462639/200928228-5d2cd6e3-0ed4-49d5-9b34-7e837449e14f.png"></code>
  <code><img title="Visual Studio Code" height="25" src="https://user-images.githubusercontent.com/26462639/200928281-73a8d46d-eb5f-4643-afab-ee45f5b7485b.png"></code>
  <code><img title="Android" height="25" src="https://user-images.githubusercontent.com/26462639/200928390-6c562f9a-db22-4214-b38c-65dfa9003953.png"></code>
  <code><img title="GitHub" height="25" src="https://user-images.githubusercontent.com/26462639/200928532-0747ab28-efd9-40f0-85ca-d68c9c1e73ff.png"></code>
  <code><img title="MySQL" height="25" src="https://user-images.githubusercontent.com/26462639/200928601-76a7e37f-1225-47f9-8a8f-c7c7d946481f.png"></code>
</p>
<hr>
<h2 align="center">📙 Dino-Run-Recurrent-PPO 📙</h2>
Reinforcement learning for chrome Dino run

<!-- ROADMAP -->
<h2 align="center">👨‍💻 Graph - Explanation 👨‍💻</h2>

it is complicated to find the right RL algorithm for a custom environment.

After many intern tests on more than 1 million samples, and with different hyperparameters for PPO it is still complicated to know where the error is and why after some time it starts to forget what it has learned, if this happens to you too, think about decreasing the learning_rate or the number of epochs (10 should be good), or increasing the clip.

Since PPO is stochastic, I advise you to change the approach by using either the DQN is DDQN variants, etc.. or define in the PPO args (deterministic=True) during the prediction of actions.

This is my first project in the field of RL.

The graph is not a good reference, but shows with few samples that PPO converges faster than DQN.

<img title="Graph PPO" src="https://user-images.githubusercontent.com/26462639/203942797-61be8ca8-4dee-4196-83da-456a8074520e.PNG">

- QRDQN_2 -> Bleu -> 1.307h relative

- RecurrentPPO -> Red -> 8.159h relative
<hr>
