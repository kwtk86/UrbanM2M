from m2m_core.test import test_main
import os
os.chdir('d://zzh2022/15.convlstm/a7_cms')

test_main(2006, 2012, 6, '../data-gisa-whn',
          r'E:\15-4.gradio/trained_models/convlstm-t08_14_23-e15.pth',
          '../data-gisa-whn/prob-e20', 64, 48, 0,
          100, 0)