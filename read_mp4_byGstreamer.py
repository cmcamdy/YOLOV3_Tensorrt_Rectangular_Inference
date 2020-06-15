
from gi.repository import GObject, Gst
import sys
import gi
gi.require_version('Gst', '1.0')

# myGst = GstreamerAppSrc()
# print('sssss:',myGst)
# Gstreamer加载Demo
def main(args):
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    GObject.threads_init()
    Gst.init(None)
    pipeline = Gst.Pipeline()
    # 加载fliesrc插件
    filesrc = Gst.ElementFactory.make("filesrc", None)
    if not filesrc:
        sys.stderr.write("'filesrc' gstreamer plugin missing\n")
        sys.exit(1)

    if Gst.uri_is_valid(args[1]):
        uri = args[1]
        print(uri)
    else:
        uri = Gst.filename_to_uri(args[1])
    # 设置资源地址
    filesrc.set_property('location', uri)

    # 解码插件
    decodebin = Gst.ElementFactory.make("decodebin", None)
    if not decodebin:
        sys.stderr.write("'decodebin' gstreamer plugin missing\n")
        sys.exit(1)

    # 显示插件
    autovideosink = Gst.ElementFactory.make("autovideosink", None)
    if not autovideosink:
        sys.stderr.write("'autovideosink' gstreamer plugin missing\n")
        sys.exit(1)

    pipeline.add(filesrc)
    pipeline.add(decodebin)
    pipeline.add(autovideosink)
    filesrc.link(decodebin)
    decode.connect('pad-added', self._decode_src_created)

    # create and event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
