#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth')


# In[ ]:


get_ipython().run_line_magic('cd', '"peft-lora-sd-dreambooth"')
get_ipython().system('pip install -r requirements.txt')


# In[ ]:


get_ipython().system('python colab.py')

