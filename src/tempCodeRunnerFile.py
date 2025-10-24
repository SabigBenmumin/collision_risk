detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections=detections)

        # วาดเส้นทาง
        