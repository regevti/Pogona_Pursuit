import imageio.v2 as iio
from pathlib import Path
from db_models import ORM, Video, VideoPrediction, PoseEstimation
import time


def get_videos_ids_for_compression(sort_by_size=False):
    orm = ORM()
    videos = {}
    with orm.session() as s:
        for v in s.query(Video).filter(Video.compression_status < 1).all():
            videos[v.id] = v.path
    # get videos sizes
    if not sort_by_size:
        return list(videos.keys())

    sizes = []
    for vid_id, vid_path in videos.items():
        try:
            size = Path(vid_path).stat().st_size
        except Exception:
            size = 0
        sizes.append((vid_id, size))

    videos_ids = [v for v, _ in sorted(sizes, key=lambda x: x[1], reverse=True)]
    return videos_ids


def compress(video_db_id):
    orm = ORM()
    with orm.session() as s:
        v = s.query(Video).filter_by(id=video_db_id).first()
        assert v is not None, 'could not find video in DB'
        writer, reader = None, None
        source = Path(v.path).resolve()
        try:
            assert source.exists(), f'video does not exist'
            dest = source.with_suffix('.mp4')

            print(f'start video compression of {source}')
            t0 = time.time()
            reader = iio.get_reader(source.as_posix())
            fps = reader.get_meta_data()['fps']
            writer = iio.get_writer(dest.as_posix(), format="FFMPEG", mode="I",
                                    fps=fps, codec="libx264", quality=5,
                                    macro_block_size=8,  # to work with 1440x1080 image size
                                    ffmpeg_log_level="error")
            for im in reader:
                writer.append_data(im)
            print(f'Finished compression of {dest} in {(time.time() - t0) / 60:.1f} minutes')

            v.path = str(dest)
            v.compression_status = 1
            source.unlink()

        except Exception as exc:
            v.compression_status = 2
            print(f'Error compressing {source}; {exc}')

        finally:
            s.commit()
            if writer is not None:
                writer.close()
            if reader is not None:
                reader.close()


def main():
    orm = ORM()
    with orm.session() as s:
        for v in s.query(Video).all():
            if v.compression_status or not (isinstance(v.path, str) and v.path.endswith('.avi')):
                continue


def foo():
    for p in Path('../output/experiments/PV26').rglob('*.avi'):
        if p.with_suffix('.mp4').exists():
            p.unlink()
            print(f'deleted {p}')


def clear_missing_videos():
    orm = ORM()
    with orm.session() as s:
        for v in s.query(Video).all():
            if not Path(v.path).exists():
                print(f'deleting from DB: {v.path}')
                for vp in s.query(VideoPrediction).filter_by(video_id=v.id).all():
                    s.delete(vp)
                for pe in s.query(PoseEstimation).filter_by(video_id=v.id).all():
                    pe.video_id = None
                s.delete(v)
                s.commit()


if __name__ == "__main__":
    vids, sizes_ = get_videos_ids_for_compression(return_sizes=True)
    compress(vids[0])
    # main()
    # foo()
    # clear_missing_videos()
