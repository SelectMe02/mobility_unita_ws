import struct
import unittest

from sensor_bridge.protocol import CameraAssembler, parse_gps, parse_imu


def camera(chunk, index=0, tail=b'EI', stamp=(12, 34), padded=False):
    header = b'MOR' + struct.pack('<4i', *stamp, index, len(chunk))
    return header + (chunk.ljust(64979, b'\0') if padded else chunk) + tail


def imu(stamped=False):
    header = b'#IMUData$' + struct.pack('<4i', 80, 0, 0, 0)
    stamp = struct.pack('<2i', 12, 34) if stamped else b''
    return header + stamp + struct.pack('<10d', 1, 0, 0, 0, 2, 3, 4, 5, 6, 9.81) + b'\r\n'


class ProtocolTests(unittest.TestCase):
    def test_imu_both_versions(self):
        for stamped in (False, True):
            stamp, values = parse_imu(imu(stamped))
            self.assertEqual(stamp, (12, 34) if stamped else None)
            self.assertEqual(values[:4], (1, 0, 0, 0))
            self.assertEqual(values[4:7], (2, 3, 4))
            self.assertAlmostEqual(values[-1], 9.81)

    def test_imu_rejects_corruption(self):
        for packet in (imu()[:-1], b'bad' + imu()[3:], imu()[:-2] + b'XX',
                       imu()[:25] + struct.pack('<10d', *([0] * 10)) + b'\r\n',
                       imu()[:25] + struct.pack('<10d', *([float('nan')] * 10)) + b'\r\n'):
            with self.assertRaises(ValueError):
                parse_imu(packet)

    def test_camera_fragments_and_padding(self):
        for padded in (False, True):
            parser = CameraAssembler()
            self.assertIsNone(parser.feed(camera(b'\xff\xd8abc', tail=b'AI', padded=padded)))
            self.assertEqual(parser.feed(camera(b'def\xff\xd9', index=1, padded=padded)),
                             ((12, 34), b'\xff\xd8abcdef\xff\xd9'))

    def test_camera_loss_and_recovery(self):
        parser = CameraAssembler()
        parser.feed(camera(b'\xff\xd8abc', tail=b'AI'))
        with self.assertRaises(ValueError):
            parser.feed(camera(b'end\xff\xd9', index=2))
        self.assertEqual(parser.feed(camera(b'\xff\xd8new\xff\xd9', stamp=(13, 0)))[1],
                         b'\xff\xd8new\xff\xd9')

    def test_camera_rejects_mixed_frames_and_oversize(self):
        parser = CameraAssembler(max_bytes=8)
        parser.feed(camera(b'\xff\xd8abc', tail=b'AI'))
        with self.assertRaises(ValueError):
            parser.feed(camera(b'\xff\xd9', index=1, stamp=(13, 0)))
        with self.assertRaises(ValueError):
            parser.feed(camera(b'\xff\xd8toolong\xff\xd9'))
        self.assertIsNone(parser.feed(b'BOXignored'))

    def test_gps_checksum_and_coordinates(self):
        packet = b'$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n'
        lat, lon, alt, quality = list(parse_gps(packet))[0]
        self.assertAlmostEqual(lat, 48.1173)
        self.assertAlmostEqual(lon, 11.5166666667)
        self.assertEqual((alt, quality), (545.4, 1))
        self.assertEqual(list(parse_gps(packet.replace(b'*47', b'*00'))), [])

    def test_gps_no_fix_and_southern_hemisphere(self):
        sentence = b'$GPGGA,123519,4807.038,S,01131.000,W,1,08,0.9,545.4,M,46.9,M,,\r\n'
        result = list(parse_gps(b'$GPRMC,ignored\r\n' + sentence + b'\x00' * 20))
        self.assertLess(result[0][0], 0)
        self.assertLess(result[0][1], 0)
        self.assertEqual(list(parse_gps(sentence.replace(b',1,08', b',0,08'))), [])
        self.assertEqual(list(parse_gps(sentence.replace(b'4807.038', b'4867.038'))), [])


if __name__ == '__main__':
    unittest.main()
