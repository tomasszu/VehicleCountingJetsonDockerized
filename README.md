<h1>Dockerized Vehicle counting solution</h1>
<h2>CUDA supported</h2>

<h3>Packages used for this process</h3>

<ul>
<li> Ultralytics package for YOLO object detection</li>
<li> Supervision package for ByteTrack object tracking and counting</li>
</ul>


<h3>Main Contents</h3>
<ul>
<li> Source code that has ben containerized</li>
<li> The Dockerfiles</li>
<li> Docker images uploaded as packages</li>
</ul>

<h3>Usage</h3>

<h4>Currently only test code enabled - counting on sample videos</h4>
<p> Download the <b>vehiclecountingit2dockerized</b> package. This is a docker container</p>

<p> Launch with <code>docker run --gpus all -it vehiclecountingit2dockerized</code></p>

<p> Make sure that nvidia-docker2 is installed.</p>
<p> Install with:
<code>sudo apt-get install nvidia-docker2</code>
<code>sudo systemctl restart docker</code>
</p>

<h4>Running code and default settings</h4>

<p>The container will run <code>python3 counting_main_CUDA.py 0</code> by default.</p>
<p>Vehicle count output is found in the shared "output_files/output_cam1.txt".</p>

<p> If you'd like to test the code with other camera videos other than the default camera 1, change the "CAM" static variable to your camera number of choice in the launched container's app/counting_test_CUDA.py. Results can be found in "app/output_files/output_cam{n}.txt".</p>


<p> You run the container to execute any code with <code>docker run -it --entrypoint /bin/bash vehiclecountingit2dockerized</code>.</p>

<h4>Docker files</h4>

<p>Docker files: "Dockerfile" contain the current setting, "Dockerfile noCUDA" contains setting up the container with no cuda support</p>


<p><b>NB!</b> Running of the "main" files for counting in camera stream is not yet functional.</p>

