import javax.websocket.*;
import javax.websocket.server.ServerEndpoint;
import javax.servlet.http.HttpServlet;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@ServerEndpoint("/droneStatus")
public class DroneStatusEndpoint {

    private static List<Session> sessions = new ArrayList<>();
    private static List<Integer> droneStatus = new ArrayList<>();  // 存储来自三台 LabVIEW 的数据

    // 当新客户端连接时触发
    @OnOpen
    public void onOpen(Session session) {
        sessions.add(session);
        System.out.println("New connection: " + session.getId());
    }

    // 当客户端关闭连接时触发
    @OnClose
    public void onClose(Session session) {
        sessions.remove(session);
        System.out.println("Closed connection: " + session.getId());
    }

    // 接收客户端发送的数据
    @OnMessage
    public void onMessage(String message, Session session) {
        try {
            // 解析来自 LabVIEW 的数据（假设是一个整数，0 或 1）
            int status = Integer.parseInt(message);
            synchronized (droneStatus) {
                // 更新数据，确保只保留最新的 3 个数据
                droneStatus.add(status);
                if (droneStatus.size() > 3) {
                    droneStatus.remove(0);
                }
            }
            
            // 将更新的数据广播到所有连接的客户端
            broadcastStatus();
        } catch (NumberFormatException e) {
            System.out.println("Invalid message format: " + message);
        }
    }

    // 广播更新后的状态给所有客户端
    private void broadcastStatus() throws IOException {
        synchronized (droneStatus) {
            String statusMessage = "{\"type\": \"status_update\", \"status\": " + droneStatus + "}";
            for (Session session : sessions) {
                session.getBasicRemote().sendText(statusMessage);
            }
        }
    }

    // 处理连接错误
    @OnError
    public void onError(Session session, Throwable error) {
        error.printStackTrace();
    }
}
