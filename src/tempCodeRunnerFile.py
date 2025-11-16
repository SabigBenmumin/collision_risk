                    # nearest = find_nearest_object(i, points)
                    # if nearest is not None and speed > 0:
                    #     ttc = nearest / speed
                    #     if ttc < 2:
                    #         cv2.putText(annotated_frame, f"RISK #{tracker_id}", (point[0], point[1]-20),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    #         handle_risk_event(
                    #             annotated_frame=annotated_frame,
                    #             output_dir=OUTPUTDIR,
                    #             detections=detections,
                    #             result=result,
                    #             tracker_id=tracker_id,
                    #             points=points,
                    #             i=i,
                    #             cap=cap,
                    #             save_log=False,
                    #             save_frame=False,
                    #         )