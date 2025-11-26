<h1>How to Run the Code</h1>
<p>Create a conda environment using <code>environment.yml</code> which will automatically install all required dependencies. (Note: Use Anaconda preferably)</p>
<pre><code>conda env create -f environment.yml</code></pre>
<pre><code>pip install git+https://github.com/ElyasYassin/AlBornoLab-myosuite.git</code></pre>
<pre><code>pip install git+https://github.com/aravindr93/mjrl.git@pvr_beta_1vk</code></pre>
(or use the Anaconda GUI to import environment) <br>
<h2> If environment.yml doesn't work for whatever reason you can:</h2>
<pre><code>conda env create -n myosuite</code></pre>
<pre><code>conda install python=3.8.18</code></pre>
<pre><code>pip install -r requirement</code></pre>

<h2>Walkthrough</h2>
<ol>
  <li>Train the policy using either <strong>mjrl training</strong> (MuJoCo Reinforcement Learning module) or <strong>sb3 training</strong> (Stable Baselines 3 module).</li>
  <li>After training is done, verify the visual outcome using <strong>load policy</strong>.</li>
</ol>

<h2>Miscellaneous</h2>
<ul>
  <li><strong>Inverse Dynamics:</strong> Reproduce a movement from a CSV file that stores the intensity of each muscle actuator as a function of time.</li>
  <li><strong>Testing Models:</strong> Use this to see the composition of models, primarily how many joints and muscles the model has.</li>
</ul>
