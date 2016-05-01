# Tracking Server

This server processes any snooker videos (currently only YouTube video's are supported) and outputs results from the neural network to a messaging queue for clients to receive.

### Installation

Ensure the py-faster-rcnn submodule has been loaded and built:

```Shell
git submodule update --init --recursive
```


